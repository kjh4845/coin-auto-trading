from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai_auto_trading.ai.inference import AIDecision
from ai_auto_trading.features.snapshot import FeatureSnapshot, TimeframeFeatureSnapshot
from ai_auto_trading.runtime.paper import (
    PaperFeedEvent,
    PaperRuntimeConfig,
    PaperTradingRuntime,
)


def _tf(
    *,
    timeframe: str,
    last_open_time: int,
    last_close: float,
    ema_fast_9: float,
    ema_slow_21: float,
    rsi_14: float,
    atr_14: float,
    cumulative_vwap: float,
    roc_5: float,
) -> TimeframeFeatureSnapshot:
    return TimeframeFeatureSnapshot(
        timeframe=timeframe,
        last_open_time=last_open_time,
        candle_count=30,
        last_close=last_close,
        current_volume=120.0,
        taker_buy_ratio=0.55,
        ema_fast_9=ema_fast_9,
        ema_slow_21=ema_slow_21,
        rsi_14=rsi_14,
        atr_14=atr_14,
        cumulative_vwap=cumulative_vwap,
        swing_high_20=last_close + 5.0,
        swing_low_20=last_close - 5.0,
        roc_5=roc_5,
        volume_sma_20=100.0,
        volume_ratio_20=1.2,
    )


def _snapshot(last_open_time: int, execution_close: float, *, bullish: bool) -> FeatureSnapshot:
    if bullish:
        execution = _tf(
            timeframe="3m",
            last_open_time=last_open_time,
            last_close=execution_close,
            ema_fast_9=execution_close - 0.01,
            ema_slow_21=execution_close - 0.10,
            rsi_14=55.0,
            atr_14=0.2,
            cumulative_vwap=execution_close - 0.05,
            roc_5=0.5,
        )
        confirmation = _tf(
            timeframe="15m",
            last_open_time=last_open_time,
            last_close=execution_close + 0.5,
            ema_fast_9=execution_close + 0.5,
            ema_slow_21=execution_close + 0.2,
            rsi_14=60.0,
            atr_14=0.5,
            cumulative_vwap=execution_close + 0.3,
            roc_5=1.0,
        )
        macro = _tf(
            timeframe="1h",
            last_open_time=last_open_time,
            last_close=execution_close + 1.0,
            ema_fast_9=execution_close + 1.0,
            ema_slow_21=execution_close + 0.7,
            rsi_14=62.0,
            atr_14=1.0,
            cumulative_vwap=execution_close + 0.8,
            roc_5=1.1,
        )
    else:
        execution = _tf(
            timeframe="3m",
            last_open_time=last_open_time,
            last_close=execution_close,
            ema_fast_9=execution_close + 0.02,
            ema_slow_21=execution_close + 0.10,
            rsi_14=47.0,
            atr_14=0.2,
            cumulative_vwap=execution_close,
            roc_5=-0.5,
        )
        confirmation = _tf(
            timeframe="15m",
            last_open_time=last_open_time,
            last_close=execution_close - 0.5,
            ema_fast_9=execution_close - 0.5,
            ema_slow_21=execution_close - 0.2,
            rsi_14=40.0,
            atr_14=0.5,
            cumulative_vwap=execution_close - 0.3,
            roc_5=-1.0,
        )
        macro = _tf(
            timeframe="1h",
            last_open_time=last_open_time,
            last_close=execution_close - 1.0,
            ema_fast_9=execution_close - 1.0,
            ema_slow_21=execution_close - 0.7,
            rsi_14=38.0,
            atr_14=1.0,
            cumulative_vwap=execution_close - 0.8,
            roc_5=-1.1,
        )
    return FeatureSnapshot(
        feature_schema_version="features_v1",
        symbol="BTCUSDT",
        generated_from="contract_price",
        timeframes={"3m": execution, "15m": confirmation, "1h": macro},
    )


def _event(
    event_time_ms: int,
    execution_close: float,
    current_mark_price: float,
    *,
    bullish: bool,
    high: float,
    low: float,
) -> PaperFeedEvent:
    return PaperFeedEvent(
        event_time_ms=event_time_ms,
        snapshot=_snapshot(event_time_ms, execution_close, bullish=bullish),
        execution_candle={
            "open_time": event_time_ms,
            "close_time": event_time_ms + 179999,
            "open": execution_close - 0.01,
            "high": high,
            "low": low,
            "close": execution_close,
        },
        current_mark_price=current_mark_price,
    )


class PaperRuntimeTest(unittest.TestCase):
    class _Assistant:
        model_base = "google/gemma-4-E2B-it"

        def __init__(self, decision: AIDecision) -> None:
            self._decision = decision

        def review_entry(self, **_: object) -> AIDecision:
            return self._decision

    def test_paper_runtime_hard_stop_exit(self) -> None:
        runtime = PaperTradingRuntime(PaperRuntimeConfig(max_holding_bars=8))
        events = [
            _event(0, 100.1, 100.1, bullish=True, high=100.2, low=100.0),
            _event(180000, 99.0, 97.4, bullish=True, high=100.0, low=98.0),
        ]
        result = runtime.run_feed(events)
        self.assertFalse(result.open_position)
        self.assertEqual(len(result.trade_records), 1)
        self.assertEqual(result.trade_records[0].exit_reason, "HARD_STOP_MARK_PRICE")

    def test_paper_runtime_time_stop_exit(self) -> None:
        runtime = PaperTradingRuntime(PaperRuntimeConfig(max_holding_bars=1))
        events = [
            _event(0, 100.1, 100.1, bullish=True, high=100.2, low=100.0),
            _event(180000, 100.15, 100.15, bullish=True, high=100.18, low=100.05),
        ]
        result = runtime.run_feed(events)
        self.assertFalse(result.open_position)
        self.assertEqual(len(result.trade_records), 1)
        self.assertEqual(result.trade_records[0].exit_reason, "TIME_STOP")

    def test_paper_runtime_exports_logs(self) -> None:
        runtime = PaperTradingRuntime(PaperRuntimeConfig(max_holding_bars=1))
        events = [
            _event(0, 100.1, 100.1, bullish=True, high=100.2, low=100.0),
            _event(180000, 100.15, 100.15, bullish=True, high=100.18, low=100.05),
        ]
        runtime.run_feed(events)
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "paper_trades.jsonl"
            runtime.export_trade_logs(output_path)
            lines = output_path.read_text().strip().splitlines()
            self.assertEqual(len(lines), 1)
            payload = json.loads(lines[0])
            self.assertEqual(payload["model_base"], "rule_only")

    def test_paper_runtime_records_ai_metadata_when_gate_allows_trade(self) -> None:
        runtime = PaperTradingRuntime(
            PaperRuntimeConfig(max_holding_bars=1),
            ai_trade_assistant=self._Assistant(
                AIDecision(
                    regime="trend_up",
                    setup_quality=0.9,
                    entry_action="allow",
                    exit_action="hold",
                    confidence=0.8,
                    reason_codes=["trend_alignment"],
                )
            ),
        )
        events = [
            _event(0, 100.1, 100.1, bullish=True, high=100.2, low=100.0),
            _event(180000, 100.15, 100.15, bullish=True, high=100.18, low=100.05),
        ]
        result = runtime.run_feed(events)
        self.assertEqual(result.trade_records[0].model_base, "google/gemma-4-E2B-it")
        self.assertIsNotNone(result.trade_records[0].ai_snapshot)


if __name__ == "__main__":
    unittest.main()
