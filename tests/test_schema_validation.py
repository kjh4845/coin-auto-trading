from __future__ import annotations

import json
from pathlib import Path
import unittest

from ai_auto_trading.models import (
    PolicyVersionInfo,
    PositionState,
    TradeRecord,
)
from ai_auto_trading.schema_validation import (
    SchemaValidationError,
    load_trade_log_schema,
    validate_trade_record,
)


class SchemaValidationTest(unittest.TestCase):
    def test_schema_loads(self) -> None:
        schema = load_trade_log_schema()
        self.assertEqual(schema["title"], "Trade Log V1")

    def test_fixture_trade_record_is_valid(self) -> None:
        fixture_path = (
            Path(__file__).resolve().parent / "fixtures" / "trade_record_rule_only.json"
        )
        payload = json.loads(fixture_path.read_text())
        validate_trade_record(payload)

    def test_generated_trade_record_is_valid(self) -> None:
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
        payload = TradeRecord.from_closed_position(
            trade_id="trade-2",
            opened_at=__import__("datetime").datetime(
                2026, 4, 15, 0, 0, tzinfo=__import__("datetime").timezone.utc
            ),
            closed_at=__import__("datetime").datetime(
                2026, 4, 15, 0, 10, tzinfo=__import__("datetime").timezone.utc
            ),
            position=position,
            policy=policy,
            exit_reason="ATR_TRAIL_EXIT",
            exit_contract_price_avg=100250.0,
            exit_mark_price=100240.0,
            max_favorable_excursion_usdt=5.0,
            max_adverse_excursion_usdt=1.5,
            fees_usdt=0.8,
            slippage_usdt=0.2,
        ).to_dict()
        validate_trade_record(payload)

    def test_generated_non_btc_trade_record_is_valid(self) -> None:
        policy = PolicyVersionInfo(
            policy_version="policy_v1",
            strategy_version="strategy_v1",
            feature_schema_version="features_v1",
        )
        position = PositionState(
            side="LONG",
            quantity=0.1,
            leverage_at_entry=2.0,
            entry_contract_price_avg=3000.0,
            entry_mark_price=3000.0,
            symbol="ETHUSDT",
        )
        payload = TradeRecord.from_closed_position(
            trade_id="trade-eth-1",
            opened_at=__import__("datetime").datetime(
                2026, 4, 15, 0, 0, tzinfo=__import__("datetime").timezone.utc
            ),
            closed_at=__import__("datetime").datetime(
                2026, 4, 15, 0, 10, tzinfo=__import__("datetime").timezone.utc
            ),
            position=position,
            policy=policy,
            exit_reason="ATR_TRAIL_EXIT",
            exit_contract_price_avg=3010.0,
            exit_mark_price=3010.0,
            max_favorable_excursion_usdt=5.0,
            max_adverse_excursion_usdt=1.5,
            fees_usdt=0.8,
            slippage_usdt=0.2,
        ).to_dict()
        validate_trade_record(payload)

    def test_missing_required_key_fails_validation(self) -> None:
        fixture_path = (
            Path(__file__).resolve().parent / "fixtures" / "trade_record_rule_only.json"
        )
        payload = json.loads(fixture_path.read_text())
        payload.pop("policy_version")
        with self.assertRaises(SchemaValidationError):
            validate_trade_record(payload)


if __name__ == "__main__":
    unittest.main()
