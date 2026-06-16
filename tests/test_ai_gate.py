from __future__ import annotations

import unittest
from unittest.mock import patch

from ai_auto_trading.ai.comparison import build_ai_comparison_report
from ai_auto_trading.ai.gate import AIGateConfig, apply_ai_entry_gate
from ai_auto_trading.ai.inference import (
    AIDecision,
    AIInferenceError,
    LocalInferenceTradeAssistant,
    parse_ai_decision,
)
from ai_auto_trading.features.snapshot import FeatureSnapshot, TimeframeFeatureSnapshot
from ai_auto_trading.strategy.rule_based import SignalDecision


def _snapshot() -> FeatureSnapshot:
    timeframe = TimeframeFeatureSnapshot(
        timeframe="3m",
        last_open_time=1,
        candle_count=30,
        last_close=100.0,
        current_volume=120.0,
        taker_buy_ratio=0.55,
        ema_fast_9=99.9,
        ema_slow_21=99.5,
        rsi_14=55.0,
        atr_14=0.2,
        cumulative_vwap=99.95,
        swing_high_20=101.0,
        swing_low_20=99.0,
        roc_5=0.1,
        volume_sma_20=100.0,
        volume_ratio_20=1.2,
    )
    return FeatureSnapshot(
        feature_schema_version="features_v1",
        symbol="BTCUSDT",
        generated_from="contract_price",
        timeframes={"3m": timeframe, "15m": timeframe, "1h": timeframe},
    )


class FakeAssistant:
    def __init__(self, decision: AIDecision) -> None:
        self.decision = decision
        self.model_base = "google/gemma-4-E2B-it"

    def review_entry(self, *, snapshot: FeatureSnapshot, rule_decision: SignalDecision) -> AIDecision:
        return self.decision


class BrokenAssistant:
    model_base = "google/gemma-4-E2B-it"

    def review_entry(self, *, snapshot: FeatureSnapshot, rule_decision: SignalDecision) -> AIDecision:
        raise AIInferenceError("bad payload")


class AIGateTest(unittest.TestCase):
    def test_parse_ai_decision_accepts_strict_json(self) -> None:
        decision = parse_ai_decision(
            '{"regime":"trend_up","setup_quality":0.8,"entry_action":"allow","exit_action":"hold","confidence":0.7,"reason_codes":["trend_alignment"]}'
        )
        self.assertEqual(decision.regime, "trend_up")
        self.assertEqual(decision.entry_action, "allow")

    def test_parse_ai_decision_rejects_missing_fields(self) -> None:
        with self.assertRaises(AIInferenceError):
            parse_ai_decision('{"regime":"trend_up"}')

    def test_parse_ai_decision_normalizes_common_model_variants(self) -> None:
        decision = parse_ai_decision(
            '{"regime":"uptrend","setup_quality":3,"entry_action":"execute_long","exit_action":"none","confidence":"high","reason_codes":["long_trend_alignment"]}'
        )
        self.assertEqual(decision.regime, "trend_up")
        self.assertEqual(decision.entry_action, "allow")
        self.assertEqual(decision.exit_action, "hold")
        self.assertAlmostEqual(decision.setup_quality, 0.6)
        self.assertAlmostEqual(decision.confidence, 0.75)

    def test_parse_ai_decision_accepts_take_profit_exit_action(self) -> None:
        decision = parse_ai_decision(
            '{"regime":"trend_up","setup_quality":0.8,"entry_action":"allow","exit_action":"TAKE_PROFIT","confidence":0.7,"reason_codes":["trend_alignment"]}'
        )
        self.assertEqual(decision.exit_action, "full_exit")

    def test_parse_ai_decision_accepts_sell_stop_exit_action(self) -> None:
        decision = parse_ai_decision(
            '{"regime":"trend_up","setup_quality":0.8,"entry_action":"allow","exit_action":"SELL_STOP","confidence":0.7,"reason_codes":["trend_alignment"]}'
        )
        self.assertEqual(decision.exit_action, "tighten_stop")

    def test_parse_ai_decision_accepts_e2b_style_qualitative_scores(self) -> None:
        decision = parse_ai_decision(
            '{"regime":"Uptrend","setup_quality":"Good","entry_action":"BUY","exit_action":"TAKE_PROFIT_LOSS_MANAGEMENT","confidence":75,"reason_codes":["long_trend_alignment"]}'
        )
        self.assertEqual(decision.regime, "trend_up")
        self.assertEqual(decision.entry_action, "allow")
        self.assertEqual(decision.exit_action, "full_exit")
        self.assertAlmostEqual(decision.setup_quality, 0.7)
        self.assertAlmostEqual(decision.confidence, 0.75)

    def test_gate_vetoes_when_regime_is_negative(self) -> None:
        result = apply_ai_entry_gate(
            rule_decision=SignalDecision(action="LONG", reason_codes=["long_trend_alignment"]),
            snapshot=_snapshot(),
            assistant=FakeAssistant(
                AIDecision(
                    regime="range",
                    setup_quality=0.9,
                    entry_action="allow",
                    exit_action="hold",
                    confidence=0.8,
                    reason_codes=["range_market"],
                )
            ),
            gate_config=AIGateConfig(min_setup_quality=0.55, reduce_size_fraction=0.5),
        )
        self.assertEqual(result.action, "NO_TRADE")
        self.assertIn("ai_regime_veto", result.reason_codes)

    def test_gate_reduces_size_when_model_requests_reduce_size(self) -> None:
        result = apply_ai_entry_gate(
            rule_decision=SignalDecision(action="LONG", reason_codes=["long_trend_alignment"]),
            snapshot=_snapshot(),
            assistant=FakeAssistant(
                AIDecision(
                    regime="trend_up",
                    setup_quality=0.9,
                    entry_action="reduce_size",
                    exit_action="hold",
                    confidence=0.8,
                    reason_codes=["thin_confirmation"],
                )
            ),
            gate_config=AIGateConfig(min_setup_quality=0.55, reduce_size_fraction=0.4),
        )
        self.assertEqual(result.action, "LONG")
        self.assertAlmostEqual(result.size_multiplier, 0.4)
        self.assertEqual(result.model_base, "google/gemma-4-E2B-it")

    def test_gate_vetoes_when_model_requests_full_exit(self) -> None:
        result = apply_ai_entry_gate(
            rule_decision=SignalDecision(action="LONG", reason_codes=["long_trend_alignment"]),
            snapshot=_snapshot(),
            assistant=FakeAssistant(
                AIDecision(
                    regime="trend_up",
                    setup_quality=0.9,
                    entry_action="allow",
                    exit_action="full_exit",
                    confidence=0.7,
                    reason_codes=["distribution_risk"],
                )
            ),
            gate_config=AIGateConfig(),
        )
        self.assertEqual(result.action, "NO_TRADE")
        self.assertIn("ai_exit_action_veto", result.reason_codes)

    def test_gate_reduces_size_when_model_requests_tight_stop(self) -> None:
        result = apply_ai_entry_gate(
            rule_decision=SignalDecision(action="LONG", reason_codes=["long_trend_alignment"]),
            snapshot=_snapshot(),
            assistant=FakeAssistant(
                AIDecision(
                    regime="trend_up",
                    setup_quality=0.9,
                    entry_action="allow",
                    exit_action="tighten_stop",
                    confidence=0.8,
                    reason_codes=["fragile_breakout"],
                )
            ),
            gate_config=AIGateConfig(min_setup_quality=0.55, reduce_size_fraction=0.4),
        )
        self.assertEqual(result.action, "LONG")
        self.assertAlmostEqual(result.size_multiplier, 0.4)
        self.assertIn("ai_exit_tighten_stop", result.reason_codes)

    def test_gate_fails_closed_on_assistant_error_by_default(self) -> None:
        result = apply_ai_entry_gate(
            rule_decision=SignalDecision(action="LONG", reason_codes=["long_trend_alignment"]),
            snapshot=_snapshot(),
            assistant=BrokenAssistant(),
            gate_config=AIGateConfig(),
        )
        self.assertEqual(result.action, "NO_TRADE")
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.model_base, "google/gemma-4-E2B-it")
        self.assertIn("ai_runtime_error_veto", result.reason_codes)

    def test_gate_vetoes_when_ai_latency_exceeds_cutoff(self) -> None:
        with patch("ai_auto_trading.ai.gate.time.perf_counter", side_effect=[0.0, 25.0]):
            result = apply_ai_entry_gate(
                rule_decision=SignalDecision(action="LONG", reason_codes=["long_trend_alignment"]),
                snapshot=_snapshot(),
                assistant=FakeAssistant(
                    AIDecision(
                        regime="trend_up",
                        setup_quality=0.9,
                        entry_action="allow",
                        exit_action="hold",
                        confidence=0.8,
                        reason_codes=["trend_alignment"],
                    )
                ),
                gate_config=AIGateConfig(max_latency_ms=20_000),
            )
        self.assertEqual(result.action, "NO_TRADE")
        self.assertIn("ai_latency_veto", result.reason_codes)
        assert result.ai_snapshot is not None
        self.assertEqual(result.ai_snapshot["latency_ms"], 25000)

    def test_gate_can_fail_open_when_explicitly_configured(self) -> None:
        result = apply_ai_entry_gate(
            rule_decision=SignalDecision(action="LONG", reason_codes=["long_trend_alignment"]),
            snapshot=_snapshot(),
            assistant=BrokenAssistant(),
            gate_config=AIGateConfig(fail_open=True),
        )
        self.assertEqual(result.action, "LONG")
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.model_base, "rule_only")

    def test_local_inference_payload_includes_model_path(self) -> None:
        payloads: list[dict[str, object]] = []

        class _Transport:
            def request(
                self,
                *,
                method: str,
                url: str,
                headers: dict[str, str],
                body: bytes,
                timeout: float,
            ) -> object:
                payloads.append(__import__("json").loads(body.decode("utf-8")))
                return {
                    "choices": [
                        {
                            "message": {
                                "content": '{"regime":"trend_up","setup_quality":0.8,"entry_action":"allow","exit_action":"hold","confidence":0.7,"reason_codes":["trend_alignment"]}'
                            }
                        }
                    ]
                }

        assistant = LocalInferenceTradeAssistant(
            model_id="google/gemma-4-E2B-it",
            model_base="google/gemma-4-E2B-it",
            model_path="/tmp/gemma4-e2b/model",
            endpoint="http://127.0.0.1:8080",
            transport=_Transport(),
        )
        decision = assistant.review_entry(
            snapshot=_snapshot(),
            rule_decision=SignalDecision(action="LONG", reason_codes=["long_trend_alignment"]),
        )
        self.assertEqual(decision.entry_action, "allow")
        self.assertEqual(payloads[0]["model_path"], "/tmp/gemma4-e2b/model")

    def test_comparison_report_computes_metric_deltas(self) -> None:
        from ai_auto_trading.backtest.replay import BacktestMetrics, BacktestResult, DecisionExplanationReport

        empty_report = DecisionExplanationReport(
            recommendation="ACCEPT",
            reasons=[],
            metrics={},
            explanation="ok",
        )
        rule_only = BacktestResult(
            trade_records=[],
            metrics=BacktestMetrics(
                trades=10,
                wins=5,
                losses=5,
                win_rate=0.5,
                total_realized_pnl_usdt=100.0,
                total_fees_usdt=10.0,
                total_slippage_usdt=5.0,
                total_pnl_after_fees_usdt=90.0,
                profit_factor=1.2,
                max_drawdown_usdt=50.0,
            ),
            decision_report=empty_report,
        )
        ai_gated = BacktestResult(
            trade_records=[],
            metrics=BacktestMetrics(
                trades=8,
                wins=5,
                losses=3,
                win_rate=0.625,
                total_realized_pnl_usdt=140.0,
                total_fees_usdt=8.0,
                total_slippage_usdt=4.0,
                total_pnl_after_fees_usdt=132.0,
                profit_factor=1.5,
                max_drawdown_usdt=40.0,
            ),
            decision_report=empty_report,
        )
        report = build_ai_comparison_report(rule_only_result=rule_only, ai_gated_result=ai_gated)
        self.assertEqual(report.deltas["trades"], -2)
        self.assertGreater(report.deltas["total_pnl_after_fees_usdt"], 0)


if __name__ == "__main__":
    unittest.main()
