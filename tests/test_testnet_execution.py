from __future__ import annotations

from io import BytesIO
import unittest
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

from ai_auto_trading.execution.testnet import (
    BinanceHTTPError,
    BinanceFuturesMainnetClient,
    BinanceFuturesTestnetClient,
    ProtectionPlacementError,
    TestnetExecutionEngine,
)
from ai_auto_trading.models import OrderIntent


class FakeTransport:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
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
                "algoStatus": "NEW",
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
        decoded_body = body.decode("utf-8") if body is not None else None
        self.calls.append(
            {"method": method, "url": url, "headers": headers or {}, "body": decoded_body}
        )

        if url.endswith("/fapi/v1/time"):
            return {"serverTime": 1234567890}
        if "/fapi/v1/exchangeInfo" in url:
            return {
                "symbols": [
                    {
                        "symbol": "BTCUSDT",
                        "filters": [
                            {
                                "filterType": "PRICE_FILTER",
                                "tickSize": "0.10",
                            },
                            {
                                "filterType": "LOT_SIZE",
                                "minQty": "0.001",
                                "stepSize": "0.001",
                            },
                            {
                                "filterType": "MARKET_LOT_SIZE",
                                "minQty": "0.001",
                                "stepSize": "0.001",
                            },
                            {
                                "filterType": "MIN_NOTIONAL",
                                "notional": "100",
                            },
                        ],
                    }
                ]
            }
        if "/fapi/v1/premiumIndex" in url:
            return {"symbol": "BTCUSDT", "markPrice": "100000.0"}
        if "/fapi/v1/positionSide/dual" in url:
            if method == "GET":
                return {"dualSidePosition": self.position_mode}
            params = parse_qs(decoded_body or "")
            self.position_mode = params["dualSidePosition"][0] == "true"
            return {"code": 200, "msg": "success"}
        if "/fapi/v1/symbolConfig" in url:
            return [dict(self.symbol_config)]
        if "/fapi/v1/leverage" in url:
            params = parse_qs(decoded_body or "")
            self.symbol_config["leverage"] = params["leverage"][0]
            return {"symbol": "BTCUSDT", "leverage": int(self.symbol_config["leverage"])}
        if "/fapi/v1/marginType" in url:
            params = parse_qs(decoded_body or "")
            self.symbol_config["marginType"] = params["marginType"][0]
            return {"code": 200, "msg": "success"}
        if "/fapi/v3/positionRisk" in url:
            return list(self.positions)
        if "/fapi/v3/account" in url:
            return {"availableBalance": "100000.0"}
        if "/fapi/v1/income" in url:
            return [
                {
                    "symbol": "ETHUSDT",
                    "incomeType": "COMMISSION",
                    "income": "-0.12",
                    "asset": "USDT",
                    "time": 111,
                    "tranId": 1,
                    "tradeId": "100",
                }
            ]
        if "/fapi/v1/openOrders" in url:
            return list(self.open_orders)
        if "/fapi/v1/openAlgoOrders" in url:
            return list(self.open_algo_orders)
        if "/fapi/v1/order/test" in url:
            return {}
        if "/fapi/v1/algoOrder" in url:
            params = parse_qs(decoded_body or "")
            return {
                "algoId": 1,
                "symbol": params["symbol"][0],
                "orderType": params["type"][0],
                "triggerPrice": params["triggerPrice"][0],
                "workingType": params["workingType"][0],
                "algoStatus": "NEW",
            }
        if "/fapi/v1/order" in url:
            params = parse_qs(decoded_body or "")
            if params["type"][0] == "MARKET":
                return {
                    "status": "FILLED",
                    "symbol": params["symbol"][0],
                    "executedQty": params["quantity"][0],
                }
            return {"status": "NEW", "symbol": params["symbol"][0]}
        if "/fapi/v1/allOpenOrders" in url:
            self.open_orders = []
            return {"code": 200, "msg": "success"}
        if "/fapi/v1/algoOpenOrders" in url and method == "DELETE":
            self.open_algo_orders = []
            return {"code": 200, "msg": "success"}
        raise AssertionError(f"Unexpected URL: {url}")


class TimestampRetryTransport(FakeTransport):
    def __init__(self) -> None:
        super().__init__()
        self.position_risk_rejections = 0

    def request(
        self,
        *,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        body: bytes | None = None,
    ):
        if "/fapi/v3/positionRisk" in url and self.position_risk_rejections == 0:
            self.position_risk_rejections += 1
            self.calls.append(
                {
                    "method": method,
                    "url": url,
                    "headers": headers or {},
                    "body": body.decode("utf-8") if body is not None else None,
                }
            )
            raise HTTPError(
                url,
                400,
                "Bad Request",
                {},
                BytesIO(
                    b'{"code":-1021,"msg":"Timestamp for this request is outside of the recvWindow."}'
                ),
            )
        return super().request(method=method, url=url, headers=headers, body=body)


class RejectingOrderTransport(FakeTransport):
    def request(
        self,
        *,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        body: bytes | None = None,
    ):
        if "/fapi/v1/order" in url and "/fapi/v1/order/test" not in url and method == "POST":
            raise HTTPError(
                url,
                400,
                "Bad Request",
                {},
                BytesIO(b'{"code":-2019,"msg":"Margin is insufficient."}'),
            )
        return super().request(method=method, url=url, headers=headers, body=body)


class StopRejectingClient:
    def __init__(self) -> None:
        self.place_calls: list[dict[str, object]] = []
        self.cancel_calls: list[dict[str, object]] = []

    def get_exchange_info(self) -> dict[str, object]:
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

    def get_symbol_rules(self, *, symbol: str):
        return BinanceFuturesTestnetClient(
            api_key="k",
            api_secret="s",
            transport=FakeTransport(),
        ).get_symbol_rules(symbol=symbol)

    def get_mark_price(self, *, symbol: str) -> dict[str, object]:
        return {"symbol": symbol, "markPrice": "100000.0"}

    def test_order(self, params: dict[str, object], *, timestamp: int) -> dict[str, object]:
        return {}

    def place_order(self, params: dict[str, object], *, timestamp: int) -> dict[str, object]:
        self.place_calls.append({"params": params, "timestamp": timestamp})
        if len(self.place_calls) == 1:
            return {
                "status": "FILLED",
                "symbol": "BTCUSDT",
                "executedQty": "0.01",
            }
        if len(self.place_calls) == 2:
            raise RuntimeError("stop rejected")
        return {
            "status": "FILLED",
            "symbol": "BTCUSDT",
            "executedQty": "0.01",
        }

    def place_algo_order(self, params: dict[str, object], *, timestamp: int) -> dict[str, object]:
        self.place_calls.append({"params": params, "timestamp": timestamp})
        raise RuntimeError("stop rejected")

    def cancel_all_open_orders(self, *, symbol: str, timestamp: int) -> dict[str, object]:
        self.cancel_calls.append({"symbol": symbol, "timestamp": timestamp})
        return {"code": 200, "msg": "success"}

    def cancel_all_open_algo_orders(self, *, symbol: str, timestamp: int) -> dict[str, object]:
        self.cancel_calls.append({"symbol": symbol, "timestamp": timestamp, "algo": True})
        return {"code": 200, "msg": "success"}


class TestnetExecutionTest(unittest.TestCase):
    def _client(self) -> BinanceFuturesTestnetClient:
        return BinanceFuturesTestnetClient(
            api_key="test-key",
            api_secret="test-secret",
            transport=FakeTransport(),
        )

    def test_public_server_time(self) -> None:
        client = self._client()
        payload = client.get_server_time()
        self.assertIn("serverTime", payload)

    def test_mainnet_client_uses_mainnet_urls_and_credentials(self) -> None:
        transport = FakeTransport()
        client = BinanceFuturesMainnetClient(
            api_key="main-key",
            api_secret="main-secret",
            transport=transport,
        )
        payload = client.get_server_time()
        self.assertIn("serverTime", payload)
        self.assertEqual(client.environment_name, "mainnet")
        self.assertEqual(client.api_key, "main-key")
        self.assertTrue(transport.calls[-1]["url"].startswith("https://fapi.binance.com"))
        self.assertIn(
            "fstream.binance.com/market/ws/btcusdt@kline_1m",
            client.kline_stream_url(symbol="BTCUSDT", interval="1m"),
        )
        self.assertIn(
            "fstream.binance.com/market/ws/btcusdt@markPrice@1s",
            client.mark_price_stream_url(symbol="BTCUSDT"),
        )
        self.assertIn(
            "fstream.binance.com/private/ws?listenKey=listen-key&events=ORDER_TRADE_UPDATE/ACCOUNT_UPDATE",
            client.user_data_stream_url(listen_key="listen-key"),
        )

    def test_testnet_client_keeps_legacy_websocket_urls(self) -> None:
        client = self._client()
        self.assertIn(
            "fstream.binancefuture.com/ws/btcusdt@kline_1m",
            client.kline_stream_url(symbol="BTCUSDT", interval="1m"),
        )
        self.assertIn(
            "fstream.binancefuture.com/ws/listen-key",
            client.user_data_stream_url(listen_key="listen-key"),
        )

    def test_signed_order_uses_signature(self) -> None:
        client = self._client()
        client.place_order(
            {
                "symbol": "BTCUSDT",
                "side": "BUY",
                "type": "MARKET",
                "quantity": "0.01",
            },
            timestamp=111,
        )
        transport = client.transport  # type: ignore[assignment]
        call = transport.calls[-1]
        body = call["body"]
        self.assertIn("signature=", body)
        parsed = parse_qs(body)
        self.assertEqual(parsed["symbol"][0], "BTCUSDT")
        self.assertEqual(parsed["timestamp"][0], "111")

    def test_signed_request_retries_once_on_binance_timestamp_error(self) -> None:
        transport = TimestampRetryTransport()
        client = BinanceFuturesTestnetClient(
            api_key="test-key",
            api_secret="test-secret",
            transport=transport,
        )

        positions = client.query_position_risk(symbol="BTCUSDT", timestamp=111)

        self.assertEqual(positions, transport.positions)
        position_calls = [
            call for call in transport.calls if "/fapi/v3/positionRisk" in call["url"]
        ]
        self.assertEqual(len(position_calls), 2)
        first_query = parse_qs(urlparse(position_calls[0]["url"]).query)
        second_query = parse_qs(urlparse(position_calls[1]["url"]).query)
        self.assertEqual(first_query["timestamp"][0], "111")
        self.assertEqual(second_query["timestamp"][0], "1234567890")

    def test_signed_request_error_includes_binance_body(self) -> None:
        client = BinanceFuturesTestnetClient(
            api_key="test-key",
            api_secret="test-secret",
            transport=RejectingOrderTransport(),
        )

        with self.assertRaises(BinanceHTTPError) as ctx:
            client.place_order(
                {
                    "symbol": "BTCUSDT",
                    "side": "BUY",
                    "type": "MARKET",
                    "quantity": "0.01",
                },
                timestamp=111,
            )

        self.assertIn("-2019", str(ctx.exception))
        self.assertIn("Margin is insufficient", str(ctx.exception))

    def test_position_risk_can_query_account_without_symbol(self) -> None:
        client = self._client()

        client.query_position_risk(timestamp=222)

        transport = client.transport  # type: ignore[assignment]
        call = transport.calls[-1]
        query = parse_qs(urlparse(call["url"]).query)
        self.assertNotIn("symbol", query)
        self.assertEqual(query["timestamp"][0], "222")

    def test_income_history_uses_signed_income_endpoint(self) -> None:
        client = self._client()

        rows = client.query_income_history(
            timestamp=222,
            start_time=1000,
            end_time=2000,
            limit=1000,
        )

        self.assertEqual(rows[0]["incomeType"], "COMMISSION")
        transport = client.transport  # type: ignore[assignment]
        call = transport.calls[-1]
        query = parse_qs(urlparse(call["url"]).query)
        self.assertIn("/fapi/v1/income", call["url"])
        self.assertEqual(query["startTime"][0], "1000")
        self.assertEqual(query["endTime"][0], "2000")
        self.assertEqual(query["limit"][0], "1000")
        self.assertEqual(query["timestamp"][0], "222")

    def test_prepare_entry_and_protection_normalizes_and_preflights(self) -> None:
        client = self._client()
        engine = TestnetExecutionEngine(client)
        prepared = engine.prepare_entry_and_protection(
            entry_intent=OrderIntent(
                side="BUY",
                quantity=0.01049,
                order_type="MARKET",
                symbol="BTCUSDT",
            ),
            hard_stop_intent=OrderIntent(
                side="SELL",
                quantity=0.01049,
                order_type="STOP_MARKET",
                symbol="BTCUSDT",
                reduce_only=True,
                working_type="MARK_PRICE",
                stop_price=97500.07,
            ),
            timestamp=222,
        )
        self.assertEqual(prepared.entry_intent.quantity, 0.01)
        self.assertEqual(prepared.hard_stop_intent.quantity, 0.01)
        self.assertEqual(prepared.hard_stop_intent.stop_price, 97500.0)

        transport = client.transport  # type: ignore[assignment]
        order_test_calls = [
            call for call in transport.calls if "/fapi/v1/order/test" in call["url"]
        ]
        self.assertEqual(len(order_test_calls), 1)

    def test_place_entry_and_protection(self) -> None:
        client = self._client()
        engine = TestnetExecutionEngine(client)
        result = engine.place_entry_and_protection(
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
            timestamp=222,
            expected_leverage=2,
        )
        self.assertEqual(result.entry_order["status"], "FILLED")
        self.assertEqual(result.hard_stop_order["algoStatus"], "NEW")

    def test_place_entry_and_protection_submits_failsafe_exit_when_stop_rejected(self) -> None:
        client = StopRejectingClient()
        engine = TestnetExecutionEngine(client)  # type: ignore[arg-type]
        with self.assertRaises(ProtectionPlacementError) as ctx:
            engine.place_entry_and_protection(
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
                timestamp=555,
                skip_preflight=True,
            )

        error = ctx.exception
        self.assertEqual(error.entry_order["status"], "FILLED")
        self.assertIsNotNone(error.failsafe_exit_order)
        self.assertEqual(error.failsafe_exit_order["status"], "FILLED")
        self.assertEqual(
            client.cancel_calls,
            [
                {"symbol": "BTCUSDT", "timestamp": 555},
                {"symbol": "BTCUSDT", "timestamp": 555, "algo": True},
            ],
        )
        self.assertEqual(len(client.place_calls), 3)
        self.assertEqual(
            client.place_calls[2]["params"],
            {
                "symbol": "BTCUSDT",
                "side": "SELL",
                "type": "MARKET",
                "quantity": "0.01",
                "reduceOnly": "true",
                "newOrderRespType": "RESULT",
            },
        )

    def test_reconcile_state_ok(self) -> None:
        client = self._client()
        engine = TestnetExecutionEngine(client)
        result = engine.reconcile_state(
            symbol="BTCUSDT",
            expected_position_qty=0.01,
            expected_stop_price_mark=97500.0,
            expected_leverage=2,
            expected_margin_mode="ISOLATED",
            expected_position_mode="ONE_WAY",
            timestamp=333,
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.mismatches, [])

    def test_reconcile_state_detects_mismatch(self) -> None:
        client = self._client()
        transport = client.transport  # type: ignore[assignment]
        transport.position_mode = True
        transport.symbol_config["marginType"] = "CROSSED"
        transport.symbol_config["leverage"] = "5"
        transport.open_orders = []
        transport.open_algo_orders = []
        transport.positions = [
            {
                "symbol": "BTCUSDT",
                "positionAmt": "0.03",
                "markPrice": "100000.0",
                "unRealizedProfit": "0.0",
            }
        ]

        engine = TestnetExecutionEngine(client)
        result = engine.reconcile_state(
            symbol="BTCUSDT",
            expected_position_qty=0.02,
            expected_stop_price_mark=98000.0,
            expected_leverage=2,
            expected_margin_mode="ISOLATED",
            expected_position_mode="ONE_WAY",
            timestamp=444,
        )
        self.assertFalse(result.ok)
        self.assertIn("position_quantity_mismatch", result.mismatches)
        self.assertIn("hard_stop_order_mismatch", result.mismatches)
        self.assertIn("position_mode_mismatch", result.mismatches)
        self.assertIn("margin_mode_mismatch", result.mismatches)
        self.assertIn("leverage_mismatch", result.mismatches)

    def test_ensure_account_configuration_updates_remote_state(self) -> None:
        client = self._client()
        transport = client.transport  # type: ignore[assignment]
        transport.position_mode = True
        transport.symbol_config["marginType"] = "CROSSED"
        transport.symbol_config["leverage"] = "7"
        transport.positions = [
            {
                "symbol": "BTCUSDT",
                "positionAmt": "0",
                "markPrice": "100000.0",
                "unRealizedProfit": "0.0",
            }
        ]
        transport.open_orders = []
        transport.open_algo_orders = []

        engine = TestnetExecutionEngine(client)
        result = engine.ensure_account_configuration(
            symbol="BTCUSDT",
            expected_leverage=2,
            expected_margin_mode="ISOLATED",
            expected_position_mode="ONE_WAY",
            timestamp=777,
        )
        self.assertTrue(result.ok)
        self.assertEqual(transport.position_mode, False)
        self.assertEqual(transport.symbol_config["marginType"], "ISOLATED")
        self.assertEqual(transport.symbol_config["leverage"], "2")

    def test_ensure_account_configuration_rejects_mode_change_with_active_state(self) -> None:
        client = self._client()
        transport = client.transport  # type: ignore[assignment]
        transport.position_mode = True
        transport.symbol_config["marginType"] = "CROSSED"
        transport.symbol_config["leverage"] = "7"

        engine = TestnetExecutionEngine(client)
        with self.assertRaisesRegex(
            ValueError,
            "cannot change position mode while positions or open orders are active",
        ):
            engine.ensure_account_configuration(
                symbol="BTCUSDT",
                expected_leverage=2,
                expected_margin_mode="ISOLATED",
                expected_position_mode="ONE_WAY",
                timestamp=778,
            )


if __name__ == "__main__":
    unittest.main()
