from __future__ import annotations

from datetime import datetime, timezone
import unittest

from ai_auto_trading.models import (
    OrderIntent,
    PolicyVersionInfo,
    PositionState,
    TradeRecord,
)


class ModelsTest(unittest.TestCase):
    def test_position_state_derives_fixed_margin_contract(self) -> None:
        position = PositionState(
            side="LONG",
            quantity=0.01,
            leverage_at_entry=2.0,
            entry_contract_price_avg=100000.0,
            entry_mark_price=100000.0,
        )
        self.assertEqual(position.filled_entry_notional, 1000.0)
        self.assertEqual(position.entry_initial_margin_fixed, 500.0)
        self.assertEqual(position.hard_stop_trigger_loss_usdt, 25.0)
        self.assertEqual(position.hard_stop_trigger_price_mark, 97500.0)

    def test_rule_only_policy_defaults_are_explicit(self) -> None:
        policy = PolicyVersionInfo(
            policy_version="policy_v1",
            strategy_version="strategy_v1",
            feature_schema_version="features_v1",
        )
        self.assertEqual(policy.model_base, "rule_only")
        self.assertIsNone(policy.dataset_version)

    def test_trade_record_from_closed_position_carries_versions(self) -> None:
        policy = PolicyVersionInfo(
            policy_version="policy_v1",
            strategy_version="strategy_v1",
            feature_schema_version="features_v1",
        )
        position = PositionState(
            side="LONG",
            quantity=0.01,
            leverage_at_entry=2.0,
            entry_contract_price_avg=100000.0,
            entry_mark_price=100000.0,
        )
        record = TradeRecord.from_closed_position(
            trade_id="trade-1",
            opened_at=datetime(2026, 4, 15, 0, 0, tzinfo=timezone.utc),
            closed_at=datetime(2026, 4, 15, 0, 10, tzinfo=timezone.utc),
            position=position,
            policy=policy,
            exit_reason="ATR_TRAIL_EXIT",
            exit_contract_price_avg=100250.0,
            exit_mark_price=100240.0,
            max_favorable_excursion_usdt=5.0,
            max_adverse_excursion_usdt=1.5,
            fees_usdt=0.8,
            slippage_usdt=0.2,
        )
        payload = record.to_dict()
        self.assertEqual(payload["model_base"], "rule_only")
        self.assertIsNone(payload["dataset_version"])
        self.assertEqual(payload["hard_stop_working_type"], "MARK_PRICE")

    def test_order_intent_accepts_stop_market_shape(self) -> None:
        intent = OrderIntent(
            side="SELL",
            quantity=0.01,
            order_type="STOP_MARKET",
            reduce_only=True,
            working_type="MARK_PRICE",
            stop_price=97500.0,
        )
        self.assertEqual(intent.order_type, "STOP_MARKET")


if __name__ == "__main__":
    unittest.main()

