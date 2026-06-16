from __future__ import annotations

import unittest

from ai_auto_trading.features.indicators import (
    atr,
    cumulative_vwap,
    ema,
    rate_of_change,
    rsi,
    swing_high,
    swing_low,
)
from ai_auto_trading.features.snapshot import FEATURE_SCHEMA_VERSION, FeatureBuilder


def _make_candle(index: int, close: float) -> dict[str, float | int]:
    return {
        "open_time": index * 60_000,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": 10.0 + index,
    }


class FeatureIndicatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.candles = [_make_candle(i, 100.0 + i) for i in range(30)]
        self.close_values = [float(c["close"]) for c in self.candles]

    def test_ema_returns_value_after_period(self) -> None:
        value = ema(self.close_values, 9)
        self.assertIsNotNone(value)
        self.assertGreater(value, 0)

    def test_rsi_returns_value_after_period(self) -> None:
        value = rsi(self.close_values, 14)
        self.assertIsNotNone(value)
        self.assertGreaterEqual(value, 0)
        self.assertLessEqual(value, 100)

    def test_atr_returns_value(self) -> None:
        value = atr(self.candles, 14)
        self.assertIsNotNone(value)
        self.assertGreater(value, 0)

    def test_vwap_and_swings(self) -> None:
        self.assertIsNotNone(cumulative_vwap(self.candles))
        self.assertEqual(swing_high(self.candles, 20), max(c["high"] for c in self.candles[-20:]))
        self.assertEqual(swing_low(self.candles, 20), min(c["low"] for c in self.candles[-20:]))

    def test_rate_of_change(self) -> None:
        value = rate_of_change(self.close_values, 5)
        self.assertIsNotNone(value)


class FeatureSnapshotTest(unittest.TestCase):
    def test_multi_timeframe_snapshot_has_version_and_timeframes(self) -> None:
        builder = FeatureBuilder()
        candles_3m = [_make_candle(i, 100.0 + i) for i in range(30)]
        candles_15m = [_make_candle(i, 200.0 + i) for i in range(30)]
        snapshot = builder.build_multi_timeframe_snapshot(
            symbol="BTCUSDT",
            candles_by_timeframe={"3m": candles_3m, "15m": candles_15m},
        )
        payload = snapshot.to_dict()
        self.assertEqual(payload["feature_schema_version"], FEATURE_SCHEMA_VERSION)
        self.assertIn("3m", payload["timeframes"])
        self.assertIn("15m", payload["timeframes"])
        self.assertEqual(payload["generated_from"], "contract_price")


if __name__ == "__main__":
    unittest.main()
