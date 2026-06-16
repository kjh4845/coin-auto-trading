from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_DOWN
import hashlib
import hmac
from io import BytesIO
import json
from typing import Any, Protocol
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from ai_auto_trading.models import OrderIntent
from ai_auto_trading.settings import Settings, load_settings, validate_leverage


class HttpTransport(Protocol):
    def request(
        self,
        *,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        body: bytes | None = None,
    ) -> Any: ...


class UrllibTransport:
    def request(
        self,
        *,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        body: bytes | None = None,
    ) -> Any:
        req = Request(url=url, data=body, method=method, headers=headers or {})
        with urlopen(req, timeout=30) as response:
            payload = response.read().decode("utf-8")
            return json.loads(payload) if payload else {}


@dataclass(frozen=True)
class ExchangeSymbolRules:
    symbol: str
    price_tick: Decimal
    quantity_step: Decimal
    min_quantity: Decimal
    market_quantity_step: Decimal
    market_min_quantity: Decimal
    min_notional: Decimal | None


@dataclass(frozen=True)
class RemoteExecutionState:
    positions: list[dict[str, Any]]
    open_orders: list[dict[str, Any]]
    open_algo_orders: list[dict[str, Any]]
    position_mode: str | None = None
    symbol_config: dict[str, Any] | None = None
    symbol_rules: ExchangeSymbolRules | None = None


@dataclass(frozen=True)
class ReconciliationResult:
    ok: bool
    mismatches: list[str]
    remote_state: RemoteExecutionState


@dataclass(frozen=True)
class ExecutionBundleResult:
    entry_order: dict[str, Any]
    hard_stop_order: dict[str, Any]


@dataclass(frozen=True)
class PreparedOrderBundle:
    symbol: str
    reference_mark_price: float
    symbol_rules: ExchangeSymbolRules
    entry_intent: OrderIntent
    hard_stop_intent: OrderIntent


@dataclass(frozen=True)
class MarginAvailability:
    ok: bool
    available_balance_usdt: float | None
    required_initial_margin_usdt: float
    entry_notional_usdt: float
    leverage: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "available_balance_usdt": self.available_balance_usdt,
            "required_initial_margin_usdt": self.required_initial_margin_usdt,
            "entry_notional_usdt": self.entry_notional_usdt,
            "leverage": self.leverage,
        }


class ExecutionConstraintError(ValueError):
    pass


class BinanceHTTPError(HTTPError):
    def __init__(self, exc: HTTPError, body: bytes) -> None:
        super().__init__(exc.url, exc.code, exc.reason, exc.headers, BytesIO(body))
        self.body = body

    @property
    def body_text(self) -> str:
        return self.body.decode("utf-8", errors="replace")

    def __str__(self) -> str:
        body_text = self.body_text.strip()
        if not body_text:
            return super().__str__()
        return f"{super().__str__()}; body={body_text}"


class ProtectionPlacementError(RuntimeError):
    def __init__(
        self,
        *,
        entry_order: dict[str, Any],
        protection_error: Exception,
        cancel_open_orders_response: dict[str, Any] | None = None,
        failsafe_exit_order: dict[str, Any] | None = None,
        cleanup_error: Exception | None = None,
    ) -> None:
        details = ["protective stop placement failed after entry"]
        if failsafe_exit_order is not None:
            details.append("failsafe exit submitted")
        if cleanup_error is not None:
            details.append(f"cleanup_error={cleanup_error}")
        super().__init__("; ".join(details))
        self.entry_order = entry_order
        self.protection_error = protection_error
        self.cancel_open_orders_response = cancel_open_orders_response
        self.failsafe_exit_order = failsafe_exit_order
        self.cleanup_error = cleanup_error


class BinanceFuturesTestnetClient:
    environment_name = "testnet"
    api_key_env_name = "BINANCE_TESTNET_API_KEY"
    api_secret_env_name = "BINANCE_TESTNET_API_SECRET"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_secret: str | None = None,
        settings: Settings | None = None,
        transport: HttpTransport | None = None,
        recv_window: int = 5000,
    ) -> None:
        self.settings = settings or load_settings()
        self.api_key = api_key or self.settings.binance_testnet_api_key
        self.api_secret = api_secret or self.settings.binance_testnet_api_secret
        self.base_url = self.settings.binance_futures_testnet_base_url.rstrip("/")
        self.ws_base_url = self.settings.binance_futures_testnet_ws_base_url.rstrip("/")
        self.transport = transport or UrllibTransport()
        self.recv_window = recv_window

    def get_server_time(self) -> dict[str, Any]:
        return self._public_request("GET", "/fapi/v1/time")

    def start_user_data_stream(self) -> dict[str, Any]:
        return self._api_key_request("POST", "/fapi/v1/listenKey")

    def keepalive_user_data_stream(self, *, listen_key: str) -> dict[str, Any]:
        return self._api_key_request(
            "PUT",
            "/fapi/v1/listenKey",
            params={"listenKey": listen_key},
        )

    def close_user_data_stream(self, *, listen_key: str) -> dict[str, Any]:
        return self._api_key_request(
            "DELETE",
            "/fapi/v1/listenKey",
            params={"listenKey": listen_key},
        )

    def user_data_stream_url(self, *, listen_key: str) -> str:
        return f"{self.ws_base_url}/ws/{listen_key}"

    def kline_stream_url(self, *, symbol: str, interval: str) -> str:
        stream = f"{symbol.lower()}@kline_{interval}"
        return f"{self.ws_base_url}/ws/{stream}"

    def mark_price_stream_url(self, *, symbol: str, speed: str = "1s") -> str:
        stream = f"{symbol.lower()}@markPrice@1s" if speed == "1s" else f"{symbol.lower()}@markPrice"
        return f"{self.ws_base_url}/ws/{stream}"

    def get_exchange_info(self) -> dict[str, Any]:
        return self._public_request("GET", "/fapi/v1/exchangeInfo")

    def get_mark_price(self, *, symbol: str) -> dict[str, Any]:
        return self._public_request("GET", "/fapi/v1/premiumIndex", params={"symbol": symbol})

    def fetch_contract_klines(
        self,
        *,
        symbol: str,
        interval: str,
        limit: int = 200,
        end_time: int | None = None,
    ) -> list[dict[str, Any]]:
        payload = self._public_request(
            "GET",
            "/fapi/v1/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit, "endTime": end_time},
        )
        rows: list[dict[str, Any]] = []
        for row in payload:
            rows.append(
                {
                    "symbol": symbol,
                    "interval": interval,
                    "open_time": int(row[0]),
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                    "close_time": int(row[6]),
                }
            )
        return rows

    def get_symbol_rules(self, *, symbol: str) -> ExchangeSymbolRules:
        exchange_info = self.get_exchange_info()
        return _symbol_rules_from_exchange_info(exchange_info, symbol)

    def test_order(self, params: dict[str, Any], *, timestamp: int) -> dict[str, Any]:
        return self._signed_request("POST", "/fapi/v1/order/test", params=params, timestamp=timestamp)

    def place_order(self, params: dict[str, Any], *, timestamp: int) -> dict[str, Any]:
        if _is_conditional_order_params(params):
            return self.place_algo_order(params, timestamp=timestamp)
        return self._signed_request("POST", "/fapi/v1/order", params=params, timestamp=timestamp)

    def place_algo_order(self, params: dict[str, Any], *, timestamp: int) -> dict[str, Any]:
        return self._signed_request(
            "POST",
            "/fapi/v1/algoOrder",
            params=_algo_order_params(params),
            timestamp=timestamp,
        )

    def query_open_orders(self, *, symbol: str, timestamp: int) -> list[dict[str, Any]]:
        return self._signed_request(
            "GET",
            "/fapi/v1/openOrders",
            params={"symbol": symbol},
            timestamp=timestamp,
        )

    def query_position_risk(self, *, symbol: str | None = None, timestamp: int) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if symbol is not None:
            params["symbol"] = symbol
        return self._signed_request(
            "GET",
            "/fapi/v3/positionRisk",
            params=params,
            timestamp=timestamp,
        )

    def query_account(self, *, timestamp: int) -> dict[str, Any]:
        return self._signed_request(
            "GET",
            "/fapi/v3/account",
            params={},
            timestamp=timestamp,
        )

    def query_income_history(
        self,
        *,
        timestamp: int,
        symbol: str | None = None,
        income_type: str | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
        page: int | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 1000))
        params = {
            "symbol": symbol,
            "incomeType": income_type,
            "startTime": start_time,
            "endTime": end_time,
            "page": page,
            "limit": limit,
        }
        return self._signed_request(
            "GET",
            "/fapi/v1/income",
            params={key: value for key, value in params.items() if value is not None},
            timestamp=timestamp,
        )

    def query_open_algo_orders(self, *, symbol: str, timestamp: int) -> list[dict[str, Any]]:
        return self._signed_request(
            "GET",
            "/fapi/v1/openAlgoOrders",
            params={"symbol": symbol},
            timestamp=timestamp,
        )

    def query_position_mode(self, *, timestamp: int) -> dict[str, Any]:
        return self._signed_request(
            "GET",
            "/fapi/v1/positionSide/dual",
            params={},
            timestamp=timestamp,
        )

    def change_position_mode(self, *, dual_side_position: bool, timestamp: int) -> dict[str, Any]:
        return self._signed_request(
            "POST",
            "/fapi/v1/positionSide/dual",
            params={"dualSidePosition": str(dual_side_position).lower()},
            timestamp=timestamp,
        )

    def query_symbol_config(self, *, symbol: str, timestamp: int) -> list[dict[str, Any]]:
        return self._signed_request(
            "GET",
            "/fapi/v1/symbolConfig",
            params={"symbol": symbol},
            timestamp=timestamp,
        )

    def change_initial_leverage(self, *, symbol: str, leverage: int, timestamp: int) -> dict[str, Any]:
        return self._signed_request(
            "POST",
            "/fapi/v1/leverage",
            params={"symbol": symbol, "leverage": leverage},
            timestamp=timestamp,
        )

    def change_margin_type(self, *, symbol: str, margin_type: str, timestamp: int) -> dict[str, Any]:
        return self._signed_request(
            "POST",
            "/fapi/v1/marginType",
            params={"symbol": symbol, "marginType": margin_type},
            timestamp=timestamp,
        )

    def cancel_all_open_orders(self, *, symbol: str, timestamp: int) -> dict[str, Any]:
        return self._signed_request(
            "DELETE",
            "/fapi/v1/allOpenOrders",
            params={"symbol": symbol},
            timestamp=timestamp,
        )

    def cancel_all_open_algo_orders(self, *, symbol: str, timestamp: int) -> dict[str, Any]:
        return self._signed_request(
            "DELETE",
            "/fapi/v1/algoOpenOrders",
            params={"symbol": symbol},
            timestamp=timestamp,
        )

    def _public_request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> Any:
        query = urlencode({k: v for k, v in (params or {}).items() if v is not None})
        url = f"{self.base_url}{path}"
        if query:
            url = f"{url}?{query}"
        return self.transport.request(method=method, url=url)

    def _api_key_request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> Any:
        if not self.api_key:
            raise ValueError(
                f"{self.environment_name} api key is required; set {self.api_key_env_name}"
            )
        query = urlencode({k: v for k, v in (params or {}).items() if v is not None})
        url = f"{self.base_url}{path}"
        if method == "GET" and query:
            url = f"{url}?{query}"
        headers = {"X-MBX-APIKEY": self.api_key}
        body = query.encode("utf-8") if method in {"POST", "PUT", "DELETE"} and query else None
        if body is not None:
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        return self.transport.request(
            method=method,
            url=url,
            headers=headers,
            body=body,
        )

    def _signed_request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any],
        timestamp: int,
    ) -> Any:
        if not self.api_key or not self.api_secret:
            raise ValueError(
                f"{self.environment_name} api credentials are required; "
                f"set {self.api_key_env_name} and {self.api_secret_env_name}"
            )

        try:
            return self._signed_request_once(
                method,
                path,
                params=params,
                timestamp=timestamp,
            )
        except HTTPError as exc:
            body = exc.read()
            if _is_binance_timestamp_error(body):
                fresh_timestamp = int(self.get_server_time()["serverTime"])
                return self._signed_request_once(
                    method,
                    path,
                    params=params,
                    timestamp=fresh_timestamp,
                )
            raise _http_error_with_body(exc, body) from exc

    def _signed_request_once(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any],
        timestamp: int,
    ) -> Any:
        signed_params = dict(params)
        signed_params["recvWindow"] = self.recv_window
        signed_params["timestamp"] = timestamp
        query = urlencode(signed_params)
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        body = f"{query}&signature={signature}".encode("utf-8")
        headers = {
            "X-MBX-APIKEY": self.api_key,
            "Content-Type": "application/x-www-form-urlencoded",
        }
        if method in {"POST", "DELETE"}:
            return self.transport.request(
                method=method,
                url=f"{self.base_url}{path}",
                headers=headers,
                body=body,
            )
        return self.transport.request(
            method=method,
            url=f"{self.base_url}{path}?{query}&signature={signature}",
            headers=headers,
        )


def _is_binance_timestamp_error(body: bytes) -> bool:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return payload.get("code") == -1021


def _http_error_with_body(exc: HTTPError, body: bytes) -> HTTPError:
    return BinanceHTTPError(exc, body)


class BinanceFuturesMainnetClient(BinanceFuturesTestnetClient):
    environment_name = "mainnet"
    api_key_env_name = "BINANCE_API_KEY"
    api_secret_env_name = "BINANCE_API_SECRET"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_secret: str | None = None,
        settings: Settings | None = None,
        transport: HttpTransport | None = None,
        recv_window: int = 5000,
    ) -> None:
        resolved_settings = settings or load_settings()
        super().__init__(
            api_key=api_key or resolved_settings.binance_api_key,
            api_secret=api_secret or resolved_settings.binance_api_secret,
            settings=resolved_settings,
            transport=transport,
            recv_window=recv_window,
        )
        self.api_key = api_key or self.settings.binance_api_key
        self.api_secret = api_secret or self.settings.binance_api_secret
        self.base_url = self.settings.binance_futures_base_url.rstrip("/")
        self.ws_base_url = self.settings.binance_futures_ws_base_url.rstrip("/")

    def _routed_ws_base_url(self, route: str) -> str:
        base_url = self.ws_base_url.rstrip("/")
        for known_route in ("/public", "/market", "/private"):
            if base_url.endswith(known_route):
                base_url = base_url[: -len(known_route)]
                break
        return f"{base_url}/{route}"

    def user_data_stream_url(self, *, listen_key: str) -> str:
        encoded_listen_key = quote(listen_key, safe="")
        return (
            f"{self._routed_ws_base_url('private')}/ws"
            f"?listenKey={encoded_listen_key}&events=ORDER_TRADE_UPDATE/ACCOUNT_UPDATE"
        )

    def kline_stream_url(self, *, symbol: str, interval: str) -> str:
        stream = f"{symbol.lower()}@kline_{interval}"
        return f"{self._routed_ws_base_url('market')}/ws/{stream}"

    def mark_price_stream_url(self, *, symbol: str, speed: str = "1s") -> str:
        stream = f"{symbol.lower()}@markPrice@1s" if speed == "1s" else f"{symbol.lower()}@markPrice"
        return f"{self._routed_ws_base_url('market')}/ws/{stream}"


class TestnetExecutionEngine:
    def __init__(self, client: BinanceFuturesTestnetClient) -> None:
        self.client = client

    def server_timestamp(self) -> int:
        payload = self.client.get_server_time()
        return int(payload["serverTime"])

    def check_entry_margin_available(
        self,
        *,
        entry_notional_usdt: float,
        leverage: int,
        timestamp: int,
    ) -> MarginAvailability:
        if leverage <= 0:
            raise ExecutionConstraintError("leverage must be > 0")
        account = self.client.query_account(timestamp=timestamp)
        available_balance = _decimal_or_none(account.get("availableBalance"))
        required_margin = _decimal(entry_notional_usdt) / _decimal(leverage)
        return MarginAvailability(
            ok=available_balance is None or available_balance >= required_margin,
            available_balance_usdt=float(available_balance) if available_balance is not None else None,
            required_initial_margin_usdt=float(required_margin),
            entry_notional_usdt=float(entry_notional_usdt),
            leverage=leverage,
        )

    def prepare_entry_and_protection(
        self,
        *,
        entry_intent: OrderIntent,
        hard_stop_intent: OrderIntent,
        timestamp: int,
        reference_mark_price: float | None = None,
        preflight: bool = True,
    ) -> PreparedOrderBundle:
        if entry_intent.symbol != hard_stop_intent.symbol:
            raise ExecutionConstraintError("entry and hard stop symbols must match")

        _validate_order_intent_shape(entry_intent)
        _validate_order_intent_shape(hard_stop_intent)

        symbol_rules = self.client.get_symbol_rules(symbol=entry_intent.symbol)
        if reference_mark_price is None:
            premium_index = self.client.get_mark_price(symbol=entry_intent.symbol)
            reference_mark_price = float(premium_index["markPrice"])

        normalized_entry = _normalize_order_intent(
            entry_intent,
            symbol_rules=symbol_rules,
            reference_price=reference_mark_price,
        )
        normalized_hard_stop = _normalize_order_intent(
            hard_stop_intent,
            symbol_rules=symbol_rules,
            reference_price=reference_mark_price,
        )

        if preflight:
            if _supports_remote_preflight(normalized_entry):
                self.client.test_order(
                    _order_intent_to_params(
                        normalized_entry,
                        new_order_resp_type="RESULT",
                    ),
                    timestamp=timestamp,
                )
            if _supports_remote_preflight(normalized_hard_stop):
                self.client.test_order(
                    _order_intent_to_params(normalized_hard_stop),
                    timestamp=timestamp,
                )

        return PreparedOrderBundle(
            symbol=entry_intent.symbol,
            reference_mark_price=reference_mark_price,
            symbol_rules=symbol_rules,
            entry_intent=normalized_entry,
            hard_stop_intent=normalized_hard_stop,
        )

    def place_entry_and_protection(
        self,
        *,
        entry_intent: OrderIntent,
        hard_stop_intent: OrderIntent,
        timestamp: int,
        expected_leverage: int | None = None,
        expected_margin_mode: str = "ISOLATED",
        expected_position_mode: str = "ONE_WAY",
        skip_preflight: bool = False,
    ) -> ExecutionBundleResult:
        if expected_leverage is not None:
            self.ensure_account_configuration(
                symbol=entry_intent.symbol,
                expected_leverage=expected_leverage,
                expected_margin_mode=expected_margin_mode,
                expected_position_mode=expected_position_mode,
                timestamp=timestamp,
            )

        prepared = self.prepare_entry_and_protection(
            entry_intent=entry_intent,
            hard_stop_intent=hard_stop_intent,
            timestamp=timestamp,
            preflight=not skip_preflight,
        )
        entry_order = self.client.place_order(
            _order_intent_to_params(
                prepared.entry_intent,
                new_order_resp_type="RESULT",
            ),
            timestamp=timestamp,
        )

        protected_quantity = _filled_quantity_from_entry(
            entry_order=entry_order,
            fallback_quantity=prepared.hard_stop_intent.quantity,
        )
        if protected_quantity is None:
            raise ExecutionConstraintError("entry order did not return a filled quantity")

        actual_hard_stop_intent = _copy_order_intent(
            prepared.hard_stop_intent,
            quantity=protected_quantity,
        )
        if not skip_preflight and protected_quantity != prepared.hard_stop_intent.quantity:
            self.client.test_order(
                _order_intent_to_params(actual_hard_stop_intent),
                timestamp=timestamp,
            )

        try:
            hard_stop_order = self.client.place_order(
                _order_intent_to_params(actual_hard_stop_intent),
                timestamp=timestamp,
            )
        except Exception as exc:
            raise self._build_protection_failure(
                entry_order=entry_order,
                entry_intent=prepared.entry_intent,
                hard_stop_intent=actual_hard_stop_intent,
                timestamp=timestamp,
                protection_error=exc,
            ) from exc

        return ExecutionBundleResult(
            entry_order=entry_order,
            hard_stop_order=hard_stop_order,
        )

    def place_failsafe_exit(
        self,
        *,
        exit_intent: OrderIntent,
        timestamp: int,
    ) -> dict[str, Any]:
        _validate_order_intent_shape(exit_intent)
        return self.client.place_order(
            _order_intent_to_params(exit_intent, new_order_resp_type="RESULT"),
            timestamp=timestamp,
        )

    def ensure_account_configuration(
        self,
        *,
        symbol: str,
        expected_leverage: int,
        expected_margin_mode: str = "ISOLATED",
        expected_position_mode: str = "ONE_WAY",
        timestamp: int,
    ) -> ReconciliationResult:
        expected_leverage = validate_leverage(expected_leverage, self.client.settings)
        current_mode = _normalize_position_mode_value(
            self.client.query_position_mode(timestamp=timestamp)
        )
        if current_mode != expected_position_mode:
            active_state = self.fetch_remote_state(symbol=symbol, timestamp=timestamp)
            if _has_active_state(active_state):
                raise ExecutionConstraintError(
                    "cannot change position mode while positions or open orders are active"
                )
            self.client.change_position_mode(
                dual_side_position=(expected_position_mode == "HEDGE"),
                timestamp=timestamp,
            )

        symbol_config = _symbol_config_for_symbol(
            self.client.query_symbol_config(symbol=symbol, timestamp=timestamp),
            symbol,
        )
        if symbol_config is None:
            raise ExecutionConstraintError(f"missing symbol config for {symbol}")

        current_margin_mode = str(symbol_config.get("marginType", "")).upper()
        if current_margin_mode != expected_margin_mode.upper():
            active_state = self.fetch_remote_state(symbol=symbol, timestamp=timestamp)
            if _has_active_state(active_state):
                raise ExecutionConstraintError(
                    "cannot change margin mode while positions or open orders are active"
                )
            self.client.change_margin_type(
                symbol=symbol,
                margin_type=expected_margin_mode.upper(),
                timestamp=timestamp,
            )

        current_leverage = _int_or_none(symbol_config.get("leverage"))
        if current_leverage != expected_leverage:
            self.client.change_initial_leverage(
                symbol=symbol,
                leverage=expected_leverage,
                timestamp=timestamp,
            )

        reconciliation = self.reconcile_state(
            symbol=symbol,
            expected_position_qty=None,
            expected_stop_price_mark=None,
            expected_leverage=expected_leverage,
            expected_margin_mode=expected_margin_mode,
            expected_position_mode=expected_position_mode,
            timestamp=timestamp,
        )
        if not reconciliation.ok:
            raise ExecutionConstraintError(
                f"account configuration mismatch: {','.join(reconciliation.mismatches)}"
            )
        return reconciliation

    def fetch_remote_state(self, *, symbol: str, timestamp: int) -> RemoteExecutionState:
        positions = self.client.query_position_risk(symbol=symbol, timestamp=timestamp)
        open_orders = self.client.query_open_orders(symbol=symbol, timestamp=timestamp)
        open_algo_orders = self.client.query_open_algo_orders(symbol=symbol, timestamp=timestamp)
        position_mode = _normalize_position_mode_value(
            self.client.query_position_mode(timestamp=timestamp)
        )
        symbol_config = _symbol_config_for_symbol(
            self.client.query_symbol_config(symbol=symbol, timestamp=timestamp),
            symbol,
        )
        symbol_rules = self.client.get_symbol_rules(symbol=symbol)
        return RemoteExecutionState(
            positions=positions,
            open_orders=open_orders,
            open_algo_orders=open_algo_orders,
            position_mode=position_mode,
            symbol_config=symbol_config,
            symbol_rules=symbol_rules,
        )

    def reconcile_state(
        self,
        *,
        symbol: str,
        expected_position_qty: float | None,
        expected_stop_price_mark: float | None,
        timestamp: int,
        expected_leverage: int | None = None,
        expected_margin_mode: str | None = None,
        expected_position_mode: str | None = None,
    ) -> ReconciliationResult:
        remote_state = self.fetch_remote_state(symbol=symbol, timestamp=timestamp)
        mismatches: list[str] = []

        remote_qty = 0.0
        for position in remote_state.positions:
            if position.get("symbol") == symbol:
                remote_qty += abs(float(position.get("positionAmt", "0")))

        if expected_position_qty is not None and abs(remote_qty - expected_position_qty) > 1e-9:
            mismatches.append("position_quantity_mismatch")

        if expected_stop_price_mark is not None:
            found_match = False
            for order in remote_state.open_orders + remote_state.open_algo_orders:
                if _matches_expected_stop_order(
                    order=order,
                    symbol=symbol,
                    expected_stop_price_mark=expected_stop_price_mark,
                ):
                    found_match = True
                    break
            if not found_match:
                mismatches.append("hard_stop_order_mismatch")

        if (
            expected_position_mode is not None
            and remote_state.position_mode != expected_position_mode
        ):
            mismatches.append("position_mode_mismatch")

        if expected_margin_mode is not None:
            remote_margin_mode = None
            if remote_state.symbol_config is not None:
                remote_margin_mode = str(
                    remote_state.symbol_config.get("marginType", "")
                ).upper()
            if remote_margin_mode != expected_margin_mode.upper():
                mismatches.append("margin_mode_mismatch")

        if expected_leverage is not None:
            remote_leverage = None
            if remote_state.symbol_config is not None:
                remote_leverage = _int_or_none(remote_state.symbol_config.get("leverage"))
            if remote_leverage != expected_leverage:
                mismatches.append("leverage_mismatch")

        return ReconciliationResult(
            ok=not mismatches,
            mismatches=mismatches,
            remote_state=remote_state,
        )

    def _build_protection_failure(
        self,
        *,
        entry_order: dict[str, Any],
        entry_intent: OrderIntent,
        hard_stop_intent: OrderIntent,
        timestamp: int,
        protection_error: Exception,
    ) -> ProtectionPlacementError:
        cancel_open_orders_response: dict[str, Any] | None = None
        cleanup_error: Exception | None = None
        failsafe_exit_order: dict[str, Any] | None = None

        try:
            cancel_open_orders_response = {
                "standard": self.client.cancel_all_open_orders(
                    symbol=entry_intent.symbol,
                    timestamp=timestamp,
                )
            }
        except Exception as exc:
            cleanup_error = exc

        try:
            algo_cancel_response = self.client.cancel_all_open_algo_orders(
                symbol=entry_intent.symbol,
                timestamp=timestamp,
            )
            cancel_open_orders_response = cancel_open_orders_response or {}
            cancel_open_orders_response["algo"] = algo_cancel_response
        except Exception as exc:
            cleanup_error = exc if cleanup_error is None else RuntimeError(
                f"{cleanup_error}; algo_cancel_error={exc}"
            )

        failsafe_exit_intent = _failsafe_exit_intent(
            entry_order=entry_order,
            hard_stop_intent=hard_stop_intent,
        )
        if failsafe_exit_intent is not None:
            try:
                failsafe_exit_order = self.place_failsafe_exit(
                    exit_intent=failsafe_exit_intent,
                    timestamp=timestamp,
                )
            except Exception as exc:
                cleanup_error = exc if cleanup_error is None else RuntimeError(
                    f"{cleanup_error}; failsafe_exit_error={exc}"
                )

        return ProtectionPlacementError(
            entry_order=entry_order,
            protection_error=protection_error,
            cancel_open_orders_response=cancel_open_orders_response,
            failsafe_exit_order=failsafe_exit_order,
            cleanup_error=cleanup_error,
        )


def _order_intent_to_params(
    intent: OrderIntent,
    *,
    new_order_resp_type: str | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "symbol": intent.symbol,
        "side": intent.side,
        "type": intent.order_type,
        "quantity": _format_decimal(_decimal(intent.quantity)),
    }
    if intent.reduce_only:
        params["reduceOnly"] = "true"
    if intent.working_type is not None:
        params["workingType"] = intent.working_type
    if intent.stop_price is not None:
        params["stopPrice"] = _format_decimal(_decimal(intent.stop_price))
    if new_order_resp_type is not None:
        params["newOrderRespType"] = new_order_resp_type
    return params


def _algo_order_params(params: dict[str, Any]) -> dict[str, Any]:
    algo_params = dict(params)
    algo_params["algoType"] = "CONDITIONAL"
    if "stopPrice" in algo_params and "triggerPrice" not in algo_params:
        algo_params["triggerPrice"] = algo_params.pop("stopPrice")
    algo_params.pop("newOrderRespType", None)
    return algo_params


def _failsafe_exit_intent(
    *, entry_order: dict[str, Any], hard_stop_intent: OrderIntent
) -> OrderIntent | None:
    quantity = _filled_quantity_from_entry(
        entry_order=entry_order,
        fallback_quantity=hard_stop_intent.quantity,
    )
    if quantity is None:
        return None
    return OrderIntent(
        side=hard_stop_intent.side,
        quantity=quantity,
        order_type="MARKET",
        symbol=hard_stop_intent.symbol,
        reduce_only=True,
    )


def _filled_quantity_from_entry(
    *, entry_order: dict[str, Any], fallback_quantity: float | None
) -> float | None:
    for key in ("executedQty", "cumQty"):
        value = entry_order.get(key)
        if value is None:
            continue
        quantity = float(value)
        if quantity > 0:
            return quantity

    if fallback_quantity is not None and str(entry_order.get("status", "")).upper() in {
        "FILLED",
        "PARTIALLY_FILLED",
    }:
        return fallback_quantity
    return None


def _validate_order_intent_shape(intent: OrderIntent) -> None:
    if intent.order_type == "STOP_MARKET":
        if intent.stop_price is None:
            raise ExecutionConstraintError("STOP_MARKET orders require stop_price")
        if intent.working_type != "MARK_PRICE":
            raise ExecutionConstraintError("STOP_MARKET protection orders must use MARK_PRICE")
    if intent.reduce_only and intent.order_type not in {"MARKET", "STOP_MARKET"}:
        raise ExecutionConstraintError("reduce_only is only supported for MARKET and STOP_MARKET")


def _normalize_order_intent(
    intent: OrderIntent,
    *,
    symbol_rules: ExchangeSymbolRules,
    reference_price: float,
) -> OrderIntent:
    quantity = _decimal(intent.quantity)
    quantity_step = symbol_rules.market_quantity_step if intent.order_type == "MARKET" else symbol_rules.quantity_step
    min_quantity = symbol_rules.market_min_quantity if intent.order_type == "MARKET" else symbol_rules.min_quantity

    normalized_quantity = _normalize_to_step(quantity, quantity_step)
    if normalized_quantity <= 0:
        raise ExecutionConstraintError("normalized quantity must be > 0")
    if normalized_quantity < min_quantity:
        raise ExecutionConstraintError("normalized quantity is below minQty")

    normalized_stop_price: float | None = None
    if intent.stop_price is not None:
        stop_price = _decimal(intent.stop_price)
        normalized_stop = _normalize_to_step(stop_price, symbol_rules.price_tick)
        if normalized_stop <= 0:
            raise ExecutionConstraintError("normalized stop price must be > 0")
        normalized_stop_price = float(normalized_stop)

    if not intent.reduce_only and symbol_rules.min_notional is not None:
        notional = _decimal(reference_price) * normalized_quantity
        if notional < symbol_rules.min_notional:
            raise ExecutionConstraintError("entry notional is below minNotional")

    return OrderIntent(
        side=intent.side,
        quantity=float(normalized_quantity),
        order_type=intent.order_type,
        symbol=intent.symbol,
        reduce_only=intent.reduce_only,
        working_type=intent.working_type,
        stop_price=normalized_stop_price,
    )


def _symbol_rules_from_exchange_info(
    exchange_info: dict[str, Any],
    symbol: str,
) -> ExchangeSymbolRules:
    symbol_info = None
    for item in exchange_info.get("symbols", []):
        if item.get("symbol") == symbol:
            symbol_info = item
            break
    if symbol_info is None:
        raise ExecutionConstraintError(f"missing exchange info for symbol {symbol}")

    filters = {
        item.get("filterType"): item
        for item in symbol_info.get("filters", [])
        if item.get("filterType")
    }
    price_filter = filters.get("PRICE_FILTER")
    lot_size = filters.get("LOT_SIZE")
    market_lot_size = filters.get("MARKET_LOT_SIZE", lot_size)
    if price_filter is None or lot_size is None or market_lot_size is None:
        raise ExecutionConstraintError(f"incomplete exchange filters for symbol {symbol}")

    min_notional_filter = filters.get("MIN_NOTIONAL") or filters.get("NOTIONAL")
    min_notional = None
    if min_notional_filter is not None:
        min_notional = _decimal_or_none(
            min_notional_filter.get("notional")
            or min_notional_filter.get("minNotional")
        )

    return ExchangeSymbolRules(
        symbol=symbol,
        price_tick=_decimal(price_filter["tickSize"]),
        quantity_step=_decimal(lot_size["stepSize"]),
        min_quantity=_decimal(lot_size["minQty"]),
        market_quantity_step=_decimal(market_lot_size["stepSize"]),
        market_min_quantity=_decimal(market_lot_size["minQty"]),
        min_notional=min_notional,
    )


def _symbol_config_for_symbol(
    symbol_configs: list[dict[str, Any]],
    symbol: str,
) -> dict[str, Any] | None:
    for item in symbol_configs:
        if item.get("symbol") == symbol:
            return item
    return None


def _normalize_to_step(value: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        return value
    units = (value / step).to_integral_value(rounding=ROUND_DOWN)
    return units * step


def _decimal(value: float | str | Decimal) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ExecutionConstraintError(f"invalid decimal value: {value!r}") from exc


def _decimal_or_none(value: Any) -> Decimal | None:
    if value in (None, ""):
        return None
    return _decimal(value)


def _format_decimal(value: Decimal) -> str:
    normalized = value.normalize()
    text = format(normalized, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _copy_order_intent(intent: OrderIntent, *, quantity: float) -> OrderIntent:
    return OrderIntent(
        side=intent.side,
        quantity=quantity,
        order_type=intent.order_type,
        symbol=intent.symbol,
        reduce_only=intent.reduce_only,
        working_type=intent.working_type,
        stop_price=intent.stop_price,
    )


def _normalize_position_mode_value(payload: dict[str, Any]) -> str:
    raw = payload.get("dualSidePosition")
    if isinstance(raw, bool):
        return "HEDGE" if raw else "ONE_WAY"
    if isinstance(raw, str):
        return "HEDGE" if raw.lower() == "true" else "ONE_WAY"
    raise ExecutionConstraintError("unexpected position mode payload")


def _int_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _supports_remote_preflight(intent: OrderIntent) -> bool:
    return intent.order_type in {"MARKET", "LIMIT"}


def _is_conditional_order_params(params: dict[str, Any]) -> bool:
    return str(params.get("type", "")).upper() in {
        "STOP",
        "STOP_MARKET",
        "TAKE_PROFIT",
        "TAKE_PROFIT_MARKET",
        "TRAILING_STOP_MARKET",
    }


def _matches_expected_stop_order(
    *,
    order: dict[str, Any],
    symbol: str,
    expected_stop_price_mark: float,
) -> bool:
    order_type = str(order.get("type") or order.get("orderType") or "").upper()
    working_type = str(order.get("workingType", "")).upper()
    trigger_price = order.get("stopPrice", order.get("triggerPrice", 0))
    return (
        str(order.get("symbol", "")).upper() == symbol.upper()
        and order_type == "STOP_MARKET"
        and working_type == "MARK_PRICE"
        and abs(float(trigger_price) - expected_stop_price_mark) < 1e-9
    )


def _has_active_state(remote_state: RemoteExecutionState) -> bool:
    if remote_state.open_orders:
        return True
    if remote_state.open_algo_orders:
        return True
    for position in remote_state.positions:
        if abs(float(position.get("positionAmt", "0"))) > 1e-9:
            return True
    return False
