from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from urllib.parse import parse_qs

from ai_auto_trading.execution.testnet import BinanceFuturesTestnetClient, TestnetExecutionEngine
from ai_auto_trading.models import OrderIntent
from ai_auto_trading.runtime.testnet import TestnetExecutionRuntime


class RuntimeFakeTransport:
    def __init__(self) -> None:
        self.position_mode = False
        self.symbol_config = {
            "symbol": "BTCUSDT",
            "marginType": "ISOLATED",
            "leverage": "2",
        }
        self.open_orders: list[dict[str, object]] = []
        self.open_algo_orders = [
            {
                "symbol": "BTCUSDT",
                "orderType": "STOP_MARKET",
                "workingType": "MARK_PRICE",
                "triggerPrice": "97500.0",
            }
        ]
        self.positions = [
            {
                "symbol": "BTCUSDT",
                "positionAmt": "0.01",
                "markPrice": "100000.0",
                "unRealizedProfit": "0.0",
            }
        ]

    def request(
        self,
        *,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        body: bytes | None = None,
    ):
        decoded_body = body.decode("utf-8") if body is not None else ""
        if url.endswith("/fapi/v1/time"):
            return {"serverTime": 1234567890}
        if "/fapi/v1/exchangeInfo" in url:
            return {
                "symbols": [
                    {
                        "symbol": "BTCUSDT",
                        "filters": [
                            {"filterType": "PRICE_FILTER", "tickSize": "0.10"},
                            {"filterType": "LOT_SIZE", "minQty": "0.001", "stepSize": "0.001"},
                            {"filterType": "MARKET_LOT_SIZE", "minQty": "0.001", "stepSize": "0.001"},
                            {"filterType": "MIN_NOTIONAL", "notional": "100"},
                        ],
                    }
                ]
            }
        if "/fapi/v1/premiumIndex" in url:
            return {"symbol": "BTCUSDT", "markPrice": "100000.0"}
        if "/fapi/v1/order/test" in url:
            return {}
        if "/fapi/v1/algoOrder" in url:
            params = parse_qs(decoded_body)
            return {
                "algoId": 1,
                "symbol": params["symbol"][0],
                "orderType": params["type"][0],
                "triggerPrice": params["triggerPrice"][0],
                "workingType": params["workingType"][0],
                "algoStatus": "NEW",
            }
        if "/fapi/v1/order" in url:
            params = parse_qs(decoded_body)
            if params["type"][0] == "MARKET":
                return {
                    "status": "FILLED",
                    "symbol": "BTCUSDT",
                    "executedQty": params["quantity"][0],
                }
            return {"status": "NEW", "symbol": "BTCUSDT"}
        if "/fapi/v3/positionRisk" in url:
            return list(self.positions)
        if "/fapi/v1/openOrders" in url:
            return list(self.open_orders)
        if "/fapi/v1/openAlgoOrders" in url and method == "GET":
            return list(self.open_algo_orders)
        if "/fapi/v1/positionSide/dual" in url:
            if method == "GET":
                return {"dualSidePosition": self.position_mode}
            return {"code": 200, "msg": "success"}
        if "/fapi/v1/symbolConfig" in url:
            return [dict(self.symbol_config)]
        if "/fapi/v1/allOpenOrders" in url:
            self.open_orders = []
            return {"code": 200, "msg": "success"}
        if "/fapi/v1/algoOpenOrders" in url and method == "DELETE":
            self.open_algo_orders = []
            return {"code": 200, "msg": "success"}
        if "/fapi/v1/leverage" in url:
            return {"symbol": "BTCUSDT", "leverage": 2}
        if "/fapi/v1/marginType" in url:
            return {"code": 200, "msg": "success"}
        raise AssertionError(f"unexpected url {url}")


class TestnetRuntimeTest(unittest.TestCase):
    def _runtime(self, db_path: Path) -> TestnetExecutionRuntime:
        client = BinanceFuturesTestnetClient(
            api_key="test-key",
            api_secret="test-secret",
            transport=RuntimeFakeTransport(),
        )
        engine = TestnetExecutionEngine(client)
        return TestnetExecutionRuntime(engine, database_path=db_path)

    def test_place_bundle_persists_expected_state_and_incident(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "runtime.sqlite3"
            runtime = self._runtime(db_path)
            result = runtime.place_entry_and_protection(
                entry_intent=OrderIntent(
                    side="BUY",
                    quantity=0.01,
                    order_type="MARKET",
                    symbol="BTCUSDT",
                ),
                hard_stop_intent=OrderIntent(
                    side="SELL",
                    quantity=0.01,
                    order_type="STOP_MARKET",
                    symbol="BTCUSDT",
                    reduce_only=True,
                    working_type="MARK_PRICE",
                    stop_price=97500.0,
                ),
                timestamp=1234567890,
                expected_leverage=2,
            )
            self.assertEqual(result.expected_state.expected_position_qty, 0.01)
            self.assertEqual(result.expected_state.expected_stop_price_mark, 97500.0)

            with sqlite3.connect(db_path) as conn:
                state_count = conn.execute(
                    "SELECT COUNT(*) FROM expected_runtime_state"
                ).fetchone()[0]
                incident_count = conn.execute(
                    "SELECT COUNT(*) FROM execution_incidents WHERE event_type = 'bundle_placed'"
                ).fetchone()[0]
            self.assertEqual(state_count, 1)
            self.assertEqual(incident_count, 1)

    def test_recover_and_reconcile_uses_stored_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "runtime.sqlite3"
            runtime = self._runtime(db_path)
            runtime.place_entry_and_protection(
                entry_intent=OrderIntent(
                    side="BUY",
                    quantity=0.01,
                    order_type="MARKET",
                    symbol="BTCUSDT",
                ),
                hard_stop_intent=OrderIntent(
                    side="SELL",
                    quantity=0.01,
                    order_type="STOP_MARKET",
                    symbol="BTCUSDT",
                    reduce_only=True,
                    working_type="MARK_PRICE",
                    stop_price=97500.0,
                ),
                timestamp=1234567890,
                expected_leverage=2,
            )
            recovered = runtime.recover_and_reconcile(
                symbol="BTCUSDT",
                timestamp=1234567891,
            )
            self.assertTrue(recovered.ok)
            self.assertIsNotNone(recovered.stored_state)
            self.assertIsNotNone(recovered.reconciliation)

            with sqlite3.connect(db_path) as conn:
                incident_count = conn.execute(
                    "SELECT COUNT(*) FROM execution_incidents WHERE event_type = 'recovery_reconciliation'"
                ).fetchone()[0]
            self.assertEqual(incident_count, 1)


if __name__ == "__main__":
    unittest.main()
