from __future__ import annotations

import unittest

from ai_auto_trading.features.snapshot import (
    FeatureSnapshot,
    TimeframeFeatureSnapshot,
)
from ai_auto_trading.strategy.rule_based import (
    RuleStrategyContext,
    RuleStrategyParameters,
    evaluate_rule_signal,
)


def _tf(
    *,
    timeframe: str,
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
        last_open_time=1,
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


class RuleBasedStrategyTest(unittest.TestCase):
    def test_long_signal(self) -> None:
        snapshot = FeatureSnapshot(
            feature_schema_version="features_v1",
            symbol="BTCUSDT",
            generated_from="contract_price",
            timeframes={
                "3m": _tf(
                    timeframe="3m",
                    last_close=100.1,
                    ema_fast_9=100.0,
                    ema_slow_21=99.5,
                    rsi_14=55.0,
                    atr_14=0.2,
                    cumulative_vwap=100.0,
                    roc_5=0.5,
                ),
                "15m": _tf(
                    timeframe="15m",
                    last_close=101.0,
                    ema_fast_9=101.0,
                    ema_slow_21=100.0,
                    rsi_14=60.0,
                    atr_14=0.5,
                    cumulative_vwap=100.8,
                    roc_5=1.0,
                ),
                "1h": _tf(
                    timeframe="1h",
                    last_close=102.0,
                    ema_fast_9=102.0,
                    ema_slow_21=101.0,
                    rsi_14=62.0,
                    atr_14=1.0,
                    cumulative_vwap=101.0,
                    roc_5=1.2,
                ),
            },
        )
        decision = evaluate_rule_signal(RuleStrategyContext(snapshot=snapshot))
        self.assertEqual(decision.action, "LONG")
        self.assertIn("long_trend_alignment", decision.reason_codes)

    def test_short_signal(self) -> None:
        snapshot = FeatureSnapshot(
            feature_schema_version="features_v1",
            symbol="BTCUSDT",
            generated_from="contract_price",
            timeframes={
                "3m": _tf(
                    timeframe="3m",
                    last_close=99.9,
                    ema_fast_9=100.0,
                    ema_slow_21=100.5,
                    rsi_14=45.0,
                    atr_14=0.2,
                    cumulative_vwap=100.0,
                    roc_5=-0.5,
                ),
                "15m": _tf(
                    timeframe="15m",
                    last_close=99.0,
                    ema_fast_9=99.0,
                    ema_slow_21=100.0,
                    rsi_14=40.0,
                    atr_14=0.5,
                    cumulative_vwap=99.2,
                    roc_5=-1.0,
                ),
                "1h": _tf(
                    timeframe="1h",
                    last_close=98.0,
                    ema_fast_9=98.0,
                    ema_slow_21=99.0,
                    rsi_14=38.0,
                    atr_14=1.0,
                    cumulative_vwap=98.8,
                    roc_5=-1.2,
                ),
            },
        )
        decision = evaluate_rule_signal(RuleStrategyContext(snapshot=snapshot))
        self.assertEqual(decision.action, "SHORT")
        self.assertIn("short_trend_alignment", decision.reason_codes)

    def test_no_trade_when_cooldown_active(self) -> None:
        snapshot = FeatureSnapshot(
            feature_schema_version="features_v1",
            symbol="BTCUSDT",
            generated_from="contract_price",
            timeframes={
                "3m": _tf(
                    timeframe="3m",
                    last_close=100.0,
                    ema_fast_9=100.0,
                    ema_slow_21=99.0,
                    rsi_14=55.0,
                    atr_14=0.2,
                    cumulative_vwap=100.0,
                    roc_5=0.2,
                ),
                "15m": _tf(
                    timeframe="15m",
                    last_close=101.0,
                    ema_fast_9=101.0,
                    ema_slow_21=100.0,
                    rsi_14=55.0,
                    atr_14=0.5,
                    cumulative_vwap=100.0,
                    roc_5=0.5,
                ),
                "1h": _tf(
                    timeframe="1h",
                    last_close=102.0,
                    ema_fast_9=102.0,
                    ema_slow_21=101.0,
                    rsi_14=55.0,
                    atr_14=1.0,
                    cumulative_vwap=101.0,
                    roc_5=0.7,
                ),
            },
        )
        decision = evaluate_rule_signal(
            RuleStrategyContext(snapshot=snapshot, cooldown_active=True)
        )
        self.assertEqual(decision.action, "NO_TRADE")
        self.assertIn("blocked_cooldown_active", decision.reason_codes)

    def test_no_trade_when_atr_too_low(self) -> None:
        snapshot = FeatureSnapshot(
            feature_schema_version="features_v1",
            symbol="BTCUSDT",
            generated_from="contract_price",
            timeframes={
                "3m": _tf(
                    timeframe="3m",
                    last_close=100.0,
                    ema_fast_9=100.0,
                    ema_slow_21=99.0,
                    rsi_14=55.0,
                    atr_14=0.001,
                    cumulative_vwap=100.0,
                    roc_5=0.2,
                ),
                "15m": _tf(
                    timeframe="15m",
                    last_close=101.0,
                    ema_fast_9=101.0,
                    ema_slow_21=100.0,
                    rsi_14=55.0,
                    atr_14=0.5,
                    cumulative_vwap=100.0,
                    roc_5=0.5,
                ),
                "1h": _tf(
                    timeframe="1h",
                    last_close=102.0,
                    ema_fast_9=102.0,
                    ema_slow_21=101.0,
                    rsi_14=55.0,
                    atr_14=1.0,
                    cumulative_vwap=101.0,
                    roc_5=0.7,
                ),
            },
        )
        decision = evaluate_rule_signal(RuleStrategyContext(snapshot=snapshot))
        self.assertEqual(decision.action, "NO_TRADE")
        self.assertIn("blocked_atr_too_low", decision.reason_codes)

    def test_no_trade_when_spread_too_wide(self) -> None:
        snapshot = FeatureSnapshot(
            feature_schema_version="features_v1",
            symbol="BTCUSDT",
            generated_from="contract_price",
            timeframes={
                "3m": _tf(
                    timeframe="3m",
                    last_close=100.0,
                    ema_fast_9=100.0,
                    ema_slow_21=99.0,
                    rsi_14=55.0,
                    atr_14=0.2,
                    cumulative_vwap=100.0,
                    roc_5=0.2,
                ),
                "15m": _tf(
                    timeframe="15m",
                    last_close=101.0,
                    ema_fast_9=101.0,
                    ema_slow_21=100.0,
                    rsi_14=55.0,
                    atr_14=0.5,
                    cumulative_vwap=100.0,
                    roc_5=0.5,
                ),
                "1h": _tf(
                    timeframe="1h",
                    last_close=102.0,
                    ema_fast_9=102.0,
                    ema_slow_21=101.0,
                    rsi_14=55.0,
                    atr_14=1.0,
                    cumulative_vwap=101.0,
                    roc_5=0.7,
                ),
            },
        )
        params = RuleStrategyParameters(max_spread_bps=1.0)
        decision = evaluate_rule_signal(
            RuleStrategyContext(snapshot=snapshot, current_spread_bps=2.0),
            params=params,
        )
        self.assertEqual(decision.action, "NO_TRADE")
        self.assertIn("blocked_spread_too_wide", decision.reason_codes)

    def test_long_signal_requires_higher_timeframe_and_volume_when_enabled(self) -> None:
        snapshot = FeatureSnapshot(
            feature_schema_version="features_v1",
            symbol="BTCUSDT",
            generated_from="contract_price",
            timeframes={
                "3m": _tf(
                    timeframe="3m",
                    last_close=100.1,
                    ema_fast_9=100.0,
                    ema_slow_21=99.5,
                    rsi_14=55.0,
                    atr_14=0.2,
                    cumulative_vwap=100.0,
                    roc_5=0.5,
                ),
                "15m": _tf(
                    timeframe="15m",
                    last_close=101.0,
                    ema_fast_9=101.0,
                    ema_slow_21=100.0,
                    rsi_14=60.0,
                    atr_14=0.5,
                    cumulative_vwap=100.8,
                    roc_5=1.0,
                ),
                "1h": _tf(
                    timeframe="1h",
                    last_close=102.0,
                    ema_fast_9=102.0,
                    ema_slow_21=101.0,
                    rsi_14=62.0,
                    atr_14=1.0,
                    cumulative_vwap=101.0,
                    roc_5=1.2,
                ),
                "4h": _tf(
                    timeframe="4h",
                    last_close=103.0,
                    ema_fast_9=103.2,
                    ema_slow_21=102.7,
                    rsi_14=63.0,
                    atr_14=1.4,
                    cumulative_vwap=102.0,
                    roc_5=1.5,
                ),
                "1d": _tf(
                    timeframe="1d",
                    last_close=104.0,
                    ema_fast_9=103.5,
                    ema_slow_21=102.9,
                    rsi_14=64.0,
                    atr_14=2.0,
                    cumulative_vwap=102.5,
                    roc_5=1.9,
                ),
            },
        )
        params = RuleStrategyParameters(
            regime_timeframe="4h",
            anchor_timeframe="1d",
            min_higher_tf_ema_spread_pct=0.001,
            min_volume_ratio_20=1.05,
        )
        decision = evaluate_rule_signal(RuleStrategyContext(snapshot=snapshot), params=params)
        self.assertEqual(decision.action, "LONG")
        self.assertIn("long_regime_alignment", decision.reason_codes)
        self.assertIn("long_volume_participation", decision.reason_codes)

    def test_long_signal_is_blocked_when_volume_participation_is_weak(self) -> None:
        execution = _tf(
            timeframe="3m",
            last_close=100.1,
            ema_fast_9=100.0,
            ema_slow_21=99.5,
            rsi_14=55.0,
            atr_14=0.2,
            cumulative_vwap=100.0,
            roc_5=0.5,
        )
        execution = TimeframeFeatureSnapshot(
            timeframe=execution.timeframe,
            last_open_time=execution.last_open_time,
            candle_count=execution.candle_count,
            last_close=execution.last_close,
            current_volume=80.0,
            taker_buy_ratio=0.4,
            ema_fast_9=execution.ema_fast_9,
            ema_slow_21=execution.ema_slow_21,
            rsi_14=execution.rsi_14,
            atr_14=execution.atr_14,
            cumulative_vwap=execution.cumulative_vwap,
            swing_high_20=execution.swing_high_20,
            swing_low_20=execution.swing_low_20,
            roc_5=execution.roc_5,
            volume_sma_20=100.0,
            volume_ratio_20=0.8,
        )
        snapshot = FeatureSnapshot(
            feature_schema_version="features_v1",
            symbol="BTCUSDT",
            generated_from="contract_price",
            timeframes={
                "3m": execution,
                "15m": _tf(
                    timeframe="15m",
                    last_close=101.0,
                    ema_fast_9=101.0,
                    ema_slow_21=100.0,
                    rsi_14=60.0,
                    atr_14=0.5,
                    cumulative_vwap=100.8,
                    roc_5=1.0,
                ),
                "1h": _tf(
                    timeframe="1h",
                    last_close=102.0,
                    ema_fast_9=102.0,
                    ema_slow_21=101.0,
                    rsi_14=62.0,
                    atr_14=1.0,
                    cumulative_vwap=101.0,
                    roc_5=1.2,
                ),
                "4h": _tf(
                    timeframe="4h",
                    last_close=103.0,
                    ema_fast_9=103.2,
                    ema_slow_21=102.7,
                    rsi_14=63.0,
                    atr_14=1.4,
                    cumulative_vwap=102.0,
                    roc_5=1.5,
                ),
                "1d": _tf(
                    timeframe="1d",
                    last_close=104.0,
                    ema_fast_9=103.5,
                    ema_slow_21=102.9,
                    rsi_14=64.0,
                    atr_14=2.0,
                    cumulative_vwap=102.5,
                    roc_5=1.9,
                ),
            },
        )
        params = RuleStrategyParameters(
            regime_timeframe="4h",
            anchor_timeframe="1d",
            min_higher_tf_ema_spread_pct=0.001,
            min_volume_ratio_20=1.05,
        )
        decision = evaluate_rule_signal(RuleStrategyContext(snapshot=snapshot), params=params)
        self.assertEqual(decision.action, "NO_TRADE")
        self.assertNotIn("long_volume_participation", decision.reason_codes)

    def test_long_signal_is_blocked_when_funding_is_overheated(self) -> None:
        snapshot = FeatureSnapshot(
            feature_schema_version="features_v1",
            symbol="BTCUSDT",
            generated_from="contract_price",
            timeframes={
                "3m": _tf(
                    timeframe="3m",
                    last_close=100.1,
                    ema_fast_9=100.0,
                    ema_slow_21=99.5,
                    rsi_14=55.0,
                    atr_14=0.2,
                    cumulative_vwap=100.0,
                    roc_5=0.5,
                ),
                "15m": _tf(
                    timeframe="15m",
                    last_close=101.0,
                    ema_fast_9=101.0,
                    ema_slow_21=100.0,
                    rsi_14=60.0,
                    atr_14=0.5,
                    cumulative_vwap=100.8,
                    roc_5=1.0,
                ),
                "1h": _tf(
                    timeframe="1h",
                    last_close=102.0,
                    ema_fast_9=102.0,
                    ema_slow_21=101.0,
                    rsi_14=62.0,
                    atr_14=1.0,
                    cumulative_vwap=101.0,
                    roc_5=1.2,
                ),
            },
        )
        params = RuleStrategyParameters(max_long_funding_rate=0.0001)
        decision = evaluate_rule_signal(
            RuleStrategyContext(snapshot=snapshot, latest_funding_rate=0.0002),
            params=params,
        )
        self.assertEqual(decision.action, "NO_TRADE")
        self.assertNotIn("long_funding_not_overheated", decision.reason_codes)

    def test_short_signal_requires_micro_bearish_trend_when_enabled(self) -> None:
        snapshot = FeatureSnapshot(
            feature_schema_version="features_v1",
            symbol="BTCUSDT",
            generated_from="contract_price",
            timeframes={
                "3m": _tf(
                    timeframe="3m",
                    last_close=99.9,
                    ema_fast_9=100.0,
                    ema_slow_21=100.5,
                    rsi_14=45.0,
                    atr_14=0.2,
                    cumulative_vwap=100.0,
                    roc_5=-0.5,
                ),
                "5m": _tf(
                    timeframe="5m",
                    last_close=100.0,
                    ema_fast_9=100.01,
                    ema_slow_21=100.0,
                    rsi_14=49.0,
                    atr_14=0.25,
                    cumulative_vwap=100.0,
                    roc_5=-0.1,
                ),
                "15m": _tf(
                    timeframe="15m",
                    last_close=99.0,
                    ema_fast_9=99.0,
                    ema_slow_21=100.0,
                    rsi_14=40.0,
                    atr_14=0.5,
                    cumulative_vwap=99.2,
                    roc_5=-1.0,
                ),
                "1h": _tf(
                    timeframe="1h",
                    last_close=98.0,
                    ema_fast_9=98.0,
                    ema_slow_21=99.0,
                    rsi_14=38.0,
                    atr_14=1.0,
                    cumulative_vwap=98.8,
                    roc_5=-1.2,
                ),
            },
        )
        params = RuleStrategyParameters(
            micro_timeframe="5m",
            min_micro_short_ema_spread_pct=0.001,
        )
        decision = evaluate_rule_signal(
            RuleStrategyContext(snapshot=snapshot),
            params=params,
        )
        self.assertEqual(decision.action, "NO_TRADE")
        self.assertNotIn("short_micro_trend_alignment", decision.reason_codes)

    def test_short_signal_requires_funding_when_filter_is_enabled(self) -> None:
        snapshot = FeatureSnapshot(
            feature_schema_version="features_v1",
            symbol="BTCUSDT",
            generated_from="contract_price",
            timeframes={
                "3m": _tf(
                    timeframe="3m",
                    last_close=99.9,
                    ema_fast_9=100.0,
                    ema_slow_21=100.5,
                    rsi_14=45.0,
                    atr_14=0.2,
                    cumulative_vwap=100.0,
                    roc_5=-0.5,
                ),
                "15m": _tf(
                    timeframe="15m",
                    last_close=99.0,
                    ema_fast_9=99.0,
                    ema_slow_21=100.0,
                    rsi_14=40.0,
                    atr_14=0.5,
                    cumulative_vwap=99.2,
                    roc_5=-1.0,
                ),
                "1h": _tf(
                    timeframe="1h",
                    last_close=98.0,
                    ema_fast_9=98.0,
                    ema_slow_21=99.0,
                    rsi_14=38.0,
                    atr_14=1.0,
                    cumulative_vwap=98.8,
                    roc_5=-1.2,
                ),
            },
        )
        params = RuleStrategyParameters(min_short_funding_rate=0.0)
        decision = evaluate_rule_signal(
            RuleStrategyContext(snapshot=snapshot),
            params=params,
        )
        self.assertEqual(decision.action, "NO_TRADE")
        self.assertIn("blocked_missing_funding_rate", decision.reason_codes)

    def test_short_signal_can_ignore_confirmation_and_momentum_conditions(self) -> None:
        snapshot = FeatureSnapshot(
            feature_schema_version="features_v1",
            symbol="BTCUSDT",
            generated_from="contract_price",
            timeframes={
                "3m": _tf(
                    timeframe="3m",
                    last_close=99.9,
                    ema_fast_9=100.0,
                    ema_slow_21=100.5,
                    rsi_14=57.0,
                    atr_14=0.2,
                    cumulative_vwap=98.0,
                    roc_5=0.8,
                ),
                "1h": _tf(
                    timeframe="1h",
                    last_close=98.0,
                    ema_fast_9=98.0,
                    ema_slow_21=99.0,
                    rsi_14=38.0,
                    atr_14=1.0,
                    cumulative_vwap=98.8,
                    roc_5=-1.2,
                ),
            },
        )
        params = RuleStrategyParameters(
            use_confirmation=False,
            use_micro=False,
            use_regime=False,
            use_anchor=False,
            require_vwap=False,
            require_rsi=False,
            require_roc=False,
        )
        decision = evaluate_rule_signal(
            RuleStrategyContext(snapshot=snapshot),
            params=params,
        )
        self.assertEqual(decision.action, "SHORT")
        self.assertIn("short_macro_alignment", decision.reason_codes)
        self.assertIn("short_price_reclaim", decision.reason_codes)

    def test_long_signal_can_ignore_macro_and_pullback_when_disabled(self) -> None:
        snapshot = FeatureSnapshot(
            feature_schema_version="features_v1",
            symbol="BTCUSDT",
            generated_from="contract_price",
            timeframes={
                "3m": _tf(
                    timeframe="3m",
                    last_close=100.1,
                    ema_fast_9=100.0,
                    ema_slow_21=99.5,
                    rsi_14=44.0,
                    atr_14=0.2,
                    cumulative_vwap=101.0,
                    roc_5=-0.2,
                ),
                "15m": _tf(
                    timeframe="15m",
                    last_close=101.0,
                    ema_fast_9=101.0,
                    ema_slow_21=100.0,
                    rsi_14=60.0,
                    atr_14=0.5,
                    cumulative_vwap=100.8,
                    roc_5=1.0,
                ),
            },
        )
        params = RuleStrategyParameters(
            use_macro=False,
            use_micro=False,
            use_regime=False,
            use_anchor=False,
            require_vwap=False,
            require_rsi=False,
            require_roc=False,
        )
        decision = evaluate_rule_signal(
            RuleStrategyContext(snapshot=snapshot),
            params=params,
        )
        self.assertEqual(decision.action, "LONG")
        self.assertIn("long_trend_alignment", decision.reason_codes)
        self.assertIn("long_price_reclaim", decision.reason_codes)


if __name__ == "__main__":
    unittest.main()
