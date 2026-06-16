from __future__ import annotations

import unittest

from ai_auto_trading.models import PositionState
from ai_auto_trading.risk.hard_stop import (
    build_exchange_hard_stop_intent,
    build_failsafe_market_exit_intent,
    evaluate_hard_stop,
    mark_price_unrealized_pnl,
)


class HardStopTest(unittest.TestCase):
    def test_long_mark_price_unrealized_pnl(self) -> None:
        position = PositionState(
            side="LONG",
            quantity=0.01,
            leverage_at_entry=2.0,
            entry_contract_price_avg=100000.0,
            entry_mark_price=100000.0,
        )
        pnl = mark_price_unrealized_pnl(position, current_mark_price=99900.0)
        self.assertEqual(pnl, -1.0)

    def test_short_mark_price_unrealized_pnl(self) -> None:
        position = PositionState(
            side="SHORT",
            quantity=0.01,
            leverage_at_entry=2.0,
            entry_contract_price_avg=100000.0,
            entry_mark_price=100000.0,
        )
        pnl = mark_price_unrealized_pnl(position, current_mark_price=99900.0)
        self.assertEqual(pnl, 1.0)

    def test_hard_stop_trigger_evaluator(self) -> None:
        position = PositionState(
            side="LONG",
            quantity=0.01,
            leverage_at_entry=2.0,
            entry_contract_price_avg=100000.0,
            entry_mark_price=100000.0,
        )
        safe = evaluate_hard_stop(position, current_mark_price=98000.0)
        breached = evaluate_hard_stop(position, current_mark_price=97499.0)
        self.assertFalse(safe.triggered)
        self.assertTrue(breached.triggered)
        self.assertEqual(breached.trigger_loss_usdt, 25.0)

    def test_hard_stop_price_matches_local_evaluator_when_entry_mark_differs(self) -> None:
        position = PositionState(
            side="LONG",
            quantity=0.01,
            leverage_at_entry=2.0,
            entry_contract_price_avg=100000.0,
            entry_mark_price=100100.0,
        )
        self.assertEqual(position.hard_stop_trigger_price_mark, 97500.0)
        evaluation = evaluate_hard_stop(
            position,
            current_mark_price=position.hard_stop_trigger_price_mark,
        )
        self.assertTrue(evaluation.triggered)
        self.assertEqual(evaluation.unrealized_pnl_usdt, -25.0)

    def test_build_exchange_hard_stop_intent(self) -> None:
        position = PositionState(
            side="LONG",
            quantity=0.01,
            leverage_at_entry=2.0,
            entry_contract_price_avg=100000.0,
            entry_mark_price=100000.0,
        )
        intent = build_exchange_hard_stop_intent(position)
        self.assertEqual(intent.side, "SELL")
        self.assertEqual(intent.order_type, "STOP_MARKET")
        self.assertTrue(intent.reduce_only)
        self.assertEqual(intent.working_type, "MARK_PRICE")
        self.assertEqual(intent.stop_price, 97500.0)

    def test_build_failsafe_market_exit_intent(self) -> None:
        position = PositionState(
            side="SHORT",
            quantity=0.02,
            leverage_at_entry=2.0,
            entry_contract_price_avg=100000.0,
            entry_mark_price=100000.0,
        )
        intent = build_failsafe_market_exit_intent(position)
        self.assertEqual(intent.side, "BUY")
        self.assertEqual(intent.order_type, "MARKET")
        self.assertTrue(intent.reduce_only)
        self.assertEqual(intent.quantity, 0.02)


if __name__ == "__main__":
    unittest.main()
