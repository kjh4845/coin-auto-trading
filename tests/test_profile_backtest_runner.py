from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from ai_auto_trading.backtest.profile_runner import (
    ProfileBacktestRun,
    _combine_trade_records,
)
from ai_auto_trading.backtest.replay import BacktestMetrics, BacktestResult, DecisionExplanationReport
from ai_auto_trading.models import PolicyVersionInfo, PositionState, TradeRecord


def _trade_record(*, trade_id: str, opened_at_ms: int, closed_at_ms: int, side: str, pnl_after_fees: float) -> TradeRecord:
    position = PositionState(
        side=side,
        quantity=1.0,
        leverage_at_entry=5.0,
        entry_contract_price_avg=100.0,
        entry_mark_price=100.0,
        symbol="BTCUSDT",
    )
    exit_price = 100.0 + pnl_after_fees if side == "LONG" else 100.0 - pnl_after_fees
    return TradeRecord.from_closed_position(
        trade_id=trade_id,
        opened_at=datetime.fromtimestamp(opened_at_ms / 1000.0, tz=timezone.utc),
        closed_at=datetime.fromtimestamp(closed_at_ms / 1000.0, tz=timezone.utc),
        position=position,
        policy=PolicyVersionInfo(
            policy_version="policy_v1",
            strategy_version="strategy_v1",
            feature_schema_version="features_v1",
        ),
        exit_reason="TIME_STOP",
        exit_contract_price_avg=exit_price,
        exit_mark_price=exit_price,
        max_favorable_excursion_usdt=max(pnl_after_fees, 0.0),
        max_adverse_excursion_usdt=0.0,
        fees_usdt=0.0,
        slippage_usdt=0.0,
        signal_reason_codes=[],
        ai_snapshot=None,
    )


def _run(profile_name: str, priority: int, trades: list[TradeRecord]) -> ProfileBacktestRun:
    empty_metrics = BacktestMetrics(
        trades=len(trades),
        wins=0,
        losses=0,
        win_rate=0.0,
        total_realized_pnl_usdt=0.0,
        total_fees_usdt=0.0,
        total_slippage_usdt=0.0,
        total_pnl_after_fees_usdt=0.0,
        profit_factor=0.0,
        max_drawdown_usdt=0.0,
    )
    return ProfileBacktestRun(
        profile_name=profile_name,
        priority=priority,
        result=BacktestResult(
            trade_records=trades,
            metrics=empty_metrics,
            decision_report=DecisionExplanationReport(
                recommendation="ACCEPT",
                reasons=[],
                metrics=empty_metrics.to_dict(),
                explanation="test",
            ),
        ),
    )


class ProfileBacktestRunnerTest(unittest.TestCase):
    def test_combine_trade_records_prefers_higher_priority_on_overlap(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        short_trade = _trade_record(
            trade_id="short-1",
            opened_at_ms=int(start.timestamp() * 1000),
            closed_at_ms=int((start + timedelta(minutes=5)).timestamp() * 1000),
            side="SHORT",
            pnl_after_fees=5.0,
        )
        long_trade = _trade_record(
            trade_id="long-1",
            opened_at_ms=int(start.timestamp() * 1000),
            closed_at_ms=int((start + timedelta(minutes=30)).timestamp() * 1000),
            side="LONG",
            pnl_after_fees=3.0,
        )
        accepted, accepted_by_profile, rejected_by_profile, exact_time_conflicts = _combine_trade_records(
            [
                _run("short_best_5m", 2, [short_trade]),
                _run("long_best_30m", 1, [long_trade]),
            ]
        )
        self.assertEqual([record.trade_id for record in accepted], ["short-1"])
        self.assertEqual(accepted_by_profile, {"short_best_5m": 1})
        self.assertEqual(rejected_by_profile, {"long_best_30m": 1})
        self.assertEqual(exact_time_conflicts, 1)

    def test_combine_trade_records_keeps_partial_legs_within_same_profile(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        partial_trade = _trade_record(
            trade_id="long-1-partial",
            opened_at_ms=int(start.timestamp() * 1000),
            closed_at_ms=int((start + timedelta(minutes=10)).timestamp() * 1000),
            side="LONG",
            pnl_after_fees=2.0,
        )
        final_trade = _trade_record(
            trade_id="long-1-final",
            opened_at_ms=int(start.timestamp() * 1000),
            closed_at_ms=int((start + timedelta(minutes=30)).timestamp() * 1000),
            side="LONG",
            pnl_after_fees=3.0,
        )
        competing_short = _trade_record(
            trade_id="short-1",
            opened_at_ms=int(start.timestamp() * 1000),
            closed_at_ms=int((start + timedelta(minutes=5)).timestamp() * 1000),
            side="SHORT",
            pnl_after_fees=1.0,
        )
        accepted, accepted_by_profile, rejected_by_profile, exact_time_conflicts = _combine_trade_records(
            [
                _run("long_best_30m", 1, [partial_trade, final_trade]),
                _run("short_best_5m", 2, [competing_short]),
            ]
        )
        self.assertEqual([record.trade_id for record in accepted], ["short-1"])
        self.assertEqual(accepted_by_profile, {"short_best_5m": 1})
        self.assertEqual(rejected_by_profile, {"long_best_30m": 1})
        self.assertEqual(exact_time_conflicts, 1)

    def test_combine_trade_records_accepts_all_partial_legs_for_single_profile(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        partial_trade = _trade_record(
            trade_id="long-1-partial",
            opened_at_ms=int(start.timestamp() * 1000),
            closed_at_ms=int((start + timedelta(minutes=10)).timestamp() * 1000),
            side="LONG",
            pnl_after_fees=2.0,
        )
        final_trade = _trade_record(
            trade_id="long-1-final",
            opened_at_ms=int(start.timestamp() * 1000),
            closed_at_ms=int((start + timedelta(minutes=30)).timestamp() * 1000),
            side="LONG",
            pnl_after_fees=3.0,
        )
        accepted, accepted_by_profile, rejected_by_profile, exact_time_conflicts = _combine_trade_records(
            [
                _run("long_best_30m", 1, [partial_trade, final_trade]),
            ]
        )
        self.assertEqual(
            [record.trade_id for record in accepted],
            ["long-1-partial", "long-1-final"],
        )
        self.assertEqual(accepted_by_profile, {"long_best_30m": 1})
        self.assertEqual(rejected_by_profile, {})
        self.assertEqual(exact_time_conflicts, 0)


if __name__ == "__main__":
    unittest.main()
