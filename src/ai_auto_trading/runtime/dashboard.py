from __future__ import annotations

from dataclasses import asdict, replace
from decimal import Decimal
import html
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import time
from typing import Any
from urllib.parse import parse_qs, urlparse

from ai_auto_trading.execution.testnet import (
    BinanceFuturesMainnetClient,
    BinanceFuturesTestnetClient,
    TestnetExecutionEngine,
)
from ai_auto_trading.runtime.orchestrator import TestnetRuntimeOrchestrator
from ai_auto_trading.runtime.testnet import TestnetExecutionRuntime, _runtime_db_connection
from ai_auto_trading.settings import Settings, load_settings
from ai_auto_trading.strategy.runtime_profiles import runtime_strategy_context

_DASHBOARD_TEMPLATE_PATH = Path(__file__).with_name("dashboard_template.html")


class DashboardController:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        symbol: str = "BTCUSDT",
        symbols: list[str] | None = None,
        environment: str = "testnet",
    ) -> None:
        self.environment = _normalize_environment(environment)
        self.symbols = _normalize_symbols(symbols or [symbol])
        self.symbol = self.symbols[0]
        resolved_settings = settings or load_settings()
        if self.environment == "mainnet" and len(self.symbols) > 1:
            resolved_settings = replace(
                resolved_settings,
                strategy_mode="multi_symbol_priority_v1",
            )
        self.settings = resolved_settings

    @property
    def runtime_database_path(self) -> Path | None:
        if self.environment == "mainnet":
            return self.settings.data_dir / "runtime" / "execution" / "mainnet_execution.sqlite3"
        return None

    def runtime(self) -> TestnetExecutionRuntime:
        return TestnetExecutionRuntime(
            None,
            settings=self.settings,
            database_path=self.runtime_database_path,
        )

    def _network_stack(
        self,
    ) -> tuple[BinanceFuturesTestnetClient, TestnetExecutionEngine, TestnetExecutionRuntime, TestnetRuntimeOrchestrator]:
        client = (
            BinanceFuturesMainnetClient(settings=self.settings)
            if self.environment == "mainnet"
            else BinanceFuturesTestnetClient(settings=self.settings)
        )
        if not client.api_key or not client.api_secret:
            if self.environment == "mainnet":
                raise ValueError(
                    "missing mainnet credentials; set BINANCE_API_KEY and BINANCE_API_SECRET"
                )
            raise ValueError(
                "missing testnet credentials; set BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_API_SECRET"
            )
        engine = TestnetExecutionEngine(client)
        runtime = TestnetExecutionRuntime(
            engine,
            settings=self.settings,
            database_path=self.runtime_database_path,
        )
        orchestrator = TestnetRuntimeOrchestrator(engine, runtime)
        return client, engine, runtime, orchestrator

    def summary(
        self,
        *,
        symbol: str | None = None,
        incident_limit: int = 10,
        display_hours: float | None = None,
    ) -> dict[str, Any]:
        runtime = self.runtime()
        target_symbol = symbol or self.symbol
        display_since_ms = _display_since_ms(display_hours)
        summary = runtime.runtime_summary(
            target_symbol,
            incident_limit=incident_limit,
            display_since_ms=display_since_ms,
        )
        account_position_symbol = _account_position_symbol(runtime, self.symbols)
        summary["account_position_symbol"] = account_position_symbol
        summary["account_position_blocked_by"] = (
            account_position_symbol
            if account_position_symbol is not None and account_position_symbol != target_symbol
            else None
        )
        summary["strategy_context"] = runtime_strategy_context(
            self.settings,
            symbols=self.symbols if len(self.symbols) > 1 else None,
        )
        return {
            "environment": self.environment,
            "symbol": target_symbol,
            "symbols": self.symbols,
            "summary": summary,
        }

    def remote_state(self, *, symbol: str | None = None) -> dict[str, Any]:
        _, engine, _, _ = self._network_stack()
        target_symbol = symbol or self.symbol
        timestamp = engine.server_timestamp()
        state = engine.fetch_remote_state(symbol=target_symbol, timestamp=timestamp)
        return {
            "environment": self.environment,
            "symbol": target_symbol,
            "symbols": self.symbols,
            "server_time_ms": timestamp,
            "remote_state": _json_ready(state),
        }

    def account_status(self) -> dict[str, Any]:
        _, engine, runtime, _ = self._network_stack()
        timestamp = engine.server_timestamp()
        account = engine.client.query_account(timestamp=timestamp)
        usdt = _account_usdt_status_from_account(account, server_time_ms=timestamp)
        runtime.record_account_snapshot(
            symbol="ACCOUNT",
            snapshot_type="rest_account_usdt_balance",
            payload={"usdt": usdt},
            recorded_at_ms=timestamp,
        )
        return {
            "environment": self.environment,
            "symbols": self.symbols,
            "server_time_ms": timestamp,
            "usdt": usdt,
        }

    def account_equity_history(
        self,
        *,
        hours: float = 168.0,
        max_points: int = 360,
    ) -> dict[str, Any]:
        runtime = self.runtime()
        runtime.initialize_storage()
        max_points = max(2, min(int(max_points), 1000))
        since_ms = None
        if hours > 0:
            since_ms = _utc_now_ms() - int(hours * 3_600_000)
        query = """
            SELECT payload_json, recorded_at_ms
            FROM account_snapshots
            WHERE symbol = 'ACCOUNT'
              AND snapshot_type = 'rest_account_usdt_balance'
        """
        params: list[Any] = []
        if since_ms is not None:
            query += " AND recorded_at_ms >= ?"
            params.append(since_ms)
        query += " ORDER BY recorded_at_ms ASC"
        with _runtime_db_connection(runtime.database_path) as conn:
            rows = conn.execute(query, params).fetchall()
        points = _account_equity_points_from_rows(rows)
        return {
            "environment": self.environment,
            "symbols": self.symbols,
            "hours": hours,
            "max_points": max_points,
            "point_count": len(points),
            "points": _downsample_points(points, max_points=max_points),
            "summary": _account_equity_summary(points),
        }

    def account_income_summary(
        self,
        *,
        hours: float = 168.0,
        max_rows: int = 1000,
        recent_limit: int = 50,
        refresh: bool = False,
    ) -> dict[str, Any]:
        runtime = self.runtime()
        end_time_ms = _utc_now_ms()
        start_time_ms = _account_income_start_time_ms(
            runtime,
            hours=hours,
            end_time_ms=end_time_ms,
        )
        max_rows = max(1, min(int(max_rows), 10_000))
        if refresh:
            client, engine, runtime, _ = self._network_stack()
            end_time_ms = engine.server_timestamp()
            start_time_ms = _account_income_start_time_ms(
                runtime,
                hours=hours,
                end_time_ms=end_time_ms,
            )
            raw_rows = _fetch_income_history(
                client,
                timestamp=end_time_ms,
                start_time_ms=start_time_ms,
                end_time_ms=end_time_ms,
                max_rows=max_rows,
            )
            runtime.persist_exchange_income_records(
                raw_rows,
                synced_at_ms=end_time_ms,
                related_trade_id=None,
            )
        rows = runtime.exchange_income_records_since(
            since_ms=start_time_ms,
            until_ms=end_time_ms,
            limit=max_rows,
        )
        recent_limit = max(0, min(int(recent_limit), 200))
        return {
            "environment": self.environment,
            "symbols": self.symbols,
            "hours": hours,
            "start_time_ms": start_time_ms,
            "end_time_ms": end_time_ms,
            "source": "stored_exchange_income_records",
            "row_count": len(rows),
            "summary": _account_income_summary(rows, symbols=self.symbols),
            "recent_income_rows": list(reversed(rows))[:recent_limit],
        }

    def service_status(self) -> dict[str, Any]:
        labels = [
            "com.ai-auto-trading.mainnet-loop",
            "com.ai-auto-trading.dashboard",
            "com.ai-auto-trading.testnet-loop",
        ]
        return {
            "environment": self.environment,
            "services": [_launchctl_status(label) for label in labels],
        }

    def handle_action(self, *, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        target_symbol = str(payload.get("symbol") or self.symbol)
        incident_limit = int(payload.get("incident_limit") or 10)
        runtime = self.runtime()
        live_exchange_actions = {
            "ensure_config",
            "monitor_once",
            "manage_position_once",
            "auto_cycle_once",
        }
        if (
            self.environment == "mainnet"
            and action in live_exchange_actions
            and not bool(payload.get("confirm_live"))
        ):
            raise ValueError(f"{action} requires live confirmation on mainnet dashboard")

        if action == "runtime_summary":
            return self.summary(symbol=target_symbol, incident_limit=incident_limit)
        if action == "remote_state":
            return self.remote_state(symbol=target_symbol)
        if action == "manual_pause":
            runtime.acquire_lockout(
                symbol=target_symbol,
                code="manual_pause",
                reason=f"paused by {self.environment} dashboard operator",
                details={"source": "dashboard", "environment": self.environment},
            )
            expected_state = runtime.load_expected_state(target_symbol)
            managed_state = runtime.load_managed_trade_state(target_symbol)
            runtime.update_runtime_status(
                symbol=target_symbol,
                state="PROTECTED" if expected_state or managed_state else "PAUSED",
            )
            return {"status": "ok", "symbol": target_symbol}
        if action == "manual_resume":
            runtime.release_lockout(
                symbol=target_symbol,
                code="manual_pause",
                reason=f"resumed by {self.environment} dashboard operator",
            )
            runtime.release_lockout(
                symbol=target_symbol,
                code="manual_review_required",
                reason=f"manual review cleared by {self.environment} dashboard operator",
            )
            runtime.release_lockout(
                symbol=target_symbol,
                code="repeated_loss_pattern_review",
                reason=f"repeated loss pattern review cleared by {self.environment} dashboard operator",
            )
            remaining = runtime.active_lockouts(target_symbol)
            expected_state = runtime.load_expected_state(target_symbol)
            managed_state = runtime.load_managed_trade_state(target_symbol)
            runtime.update_runtime_status(
                symbol=target_symbol,
                state="PROTECTED" if expected_state or managed_state else ("PAUSED" if remaining else "READY"),
            )
            return {"status": "ok", "symbol": target_symbol}
        if action == "clear_expected_state":
            runtime.clear_expected_state(target_symbol)
            runtime.clear_managed_trade_state(target_symbol)
            remaining = runtime.active_lockouts(target_symbol)
            runtime.update_runtime_status(
                symbol=target_symbol,
                state="PAUSED" if remaining else "READY",
            )
            return {"status": "ok", "symbol": target_symbol}

        client, engine, runtime, orchestrator = self._network_stack()
        timestamp = engine.server_timestamp()
        if action == "user_stream_start":
            session = orchestrator.start_user_stream(symbol=target_symbol, timestamp=timestamp)
            return {
                "status": "ok",
                "server_time_ms": timestamp,
                "session": _json_ready(session),
            }
        if action == "user_stream_keepalive":
            session = orchestrator.keepalive_user_stream(symbol=target_symbol, timestamp=timestamp)
            return {
                "status": "ok",
                "server_time_ms": timestamp,
                "session": _json_ready(session),
            }
        if action == "user_stream_close":
            orchestrator.close_user_stream(symbol=target_symbol)
            return {"status": "ok", "symbol": target_symbol}
        if action == "ensure_config":
            leverage = int(payload.get("leverage") or self.settings.live_start_leverage)
            margin_mode = str(payload.get("margin_mode") or "ISOLATED")
            position_mode = str(payload.get("position_mode") or "ONE_WAY")
            result = engine.ensure_account_configuration(
                symbol=target_symbol,
                expected_leverage=leverage,
                expected_margin_mode=margin_mode,
                expected_position_mode=position_mode,
                timestamp=timestamp,
            )
            return {
                "status": "ok",
                "server_time_ms": timestamp,
                "reconciliation": _json_ready(result),
            }
        if action == "monitor_once":
            result = orchestrator.reconcile_once(symbol=target_symbol, timestamp=timestamp)
            return {
                "status": "ok" if not result.active_lockouts else "paused",
                "server_time_ms": timestamp,
                "monitor_result": _json_ready(result),
            }
        if action == "manage_position_once":
            result = orchestrator.manage_open_position_once(
                symbol=target_symbol,
                timestamp=timestamp,
                candle_limit=int(payload.get("candle_limit") or 120),
            )
            return {
                "status": "ok" if result.action != "EXIT" else "exited",
                "server_time_ms": timestamp,
                "result": _json_ready(result),
            }
        if action == "auto_cycle_once":
            if self.environment == "mainnet" and len(self.symbols) > 1:
                return self._handle_mainnet_priority_cycle_once(payload=payload)
            monitor = orchestrator.reconcile_once(symbol=target_symbol, timestamp=timestamp)
            managed_state = runtime.load_managed_trade_state(target_symbol)
            if monitor.safety_action is not None:
                return {
                    "status": "ok" if not monitor.active_lockouts else "paused",
                    "server_time_ms": timestamp,
                    "monitor_result": _json_ready(monitor),
                    "result": None,
                }
            if managed_state is not None:
                result = orchestrator.manage_open_position_once(
                    symbol=target_symbol,
                    timestamp=timestamp,
                    candle_limit=int(payload.get("candle_limit") or 120),
                )
            else:
                if monitor.active_lockouts:
                    return {
                        "status": "ok" if not monitor.active_lockouts else "paused",
                        "server_time_ms": timestamp,
                        "monitor_result": _json_ready(monitor),
                        "result": None,
                    }
                trigger_close_time_ms = orchestrator.latest_runtime_trigger_close_time(
                    symbol=target_symbol,
                    timestamp=timestamp,
                )
                leverage = int(payload.get("leverage") or self.settings.live_start_leverage)
                sizing = None
                if self.environment == "mainnet":
                    fixed_entry_notional_usdt = _float_or_none(payload.get("entry_notional_usdt"))
                    entry_margin_fraction = float(payload.get("entry_margin_fraction") or 0.98)
                    entry_notional_usdt, sizing = _mainnet_entry_notional_usdt(
                        engine,
                        fixed_entry_notional_usdt=fixed_entry_notional_usdt,
                        leverage=leverage,
                        entry_margin_fraction=entry_margin_fraction,
                    )
                else:
                    entry_notional_usdt = float(payload.get("entry_notional_usdt") or 1000.0)
                result = orchestrator.attempt_rule_entry_once(
                    symbol=target_symbol,
                    timestamp=timestamp,
                    entry_notional_usdt=entry_notional_usdt,
                    leverage=leverage,
                    candle_limit=int(payload.get("candle_limit") or 120),
                    trigger_close_time_ms=trigger_close_time_ms,
                )
            return {
                "status": "ok",
                "server_time_ms": timestamp,
                "sizing": _json_ready(sizing),
                "monitor_result": _json_ready(monitor),
                "result": _json_ready(result),
            }
        raise ValueError(f"unknown action: {action}")

    def _handle_mainnet_priority_cycle_once(self, *, payload: dict[str, Any]) -> dict[str, Any]:
        _, engine, runtime, orchestrator = self._network_stack()
        candle_limit = int(payload.get("candle_limit") or 120)
        entry_notional_usdt = _float_or_none(payload.get("entry_notional_usdt"))
        entry_margin_fraction = float(payload.get("entry_margin_fraction") or 0.98)
        cycle_started_ms = engine.server_timestamp()
        monitors: dict[str, Any] = {}
        symbol_results: dict[str, Any] = {}

        for symbol in self.symbols:
            runtime.settings = _mainnet_priority_settings(self.settings, symbol)
            timestamp = engine.server_timestamp()
            monitors[symbol] = orchestrator.reconcile_once(symbol=symbol, timestamp=timestamp)

        managed = _managed_symbol(runtime, self.symbols)
        if managed is not None:
            runtime.settings = _mainnet_priority_settings(self.settings, managed)
            timestamp = engine.server_timestamp()
            result = orchestrator.manage_open_position_once(
                symbol=managed,
                timestamp=timestamp,
                candle_limit=candle_limit,
            )
            return {
                "status": "ok",
                "server_time_ms": timestamp,
                "cycle_started_ms": cycle_started_ms,
                "action": "MANAGE_OPEN_POSITION",
                "managed_symbol": managed,
                "monitors": _json_ready(monitors),
                "result": _json_ready(result),
            }

        remote_open_symbols = [
            symbol
            for symbol, monitor in monitors.items()
            if _remote_has_open_position(monitor)
        ]
        if remote_open_symbols:
            return {
                "status": "ok",
                "server_time_ms": engine.server_timestamp(),
                "cycle_started_ms": cycle_started_ms,
                "action": "SKIP_REMOTE_POSITION",
                "remote_open_symbols": remote_open_symbols,
                "monitors": _json_ready(monitors),
                "result": None,
            }

        account_open_symbols = _remote_account_open_position_symbols(engine)
        if account_open_symbols:
            return {
                "status": "ok",
                "server_time_ms": engine.server_timestamp(),
                "cycle_started_ms": cycle_started_ms,
                "action": "SKIP_ACCOUNT_POSITION",
                "remote_open_symbols": account_open_symbols,
                "monitors": _json_ready(monitors),
                "result": None,
            }

        for symbol in self.symbols:
            monitor = monitors[symbol]
            if monitor.safety_action is not None:
                symbol_results[symbol] = {
                    "action": "SKIP_SAFETY_ACTION",
                    "safety_action": monitor.safety_action,
                }
                continue
            if monitor.active_lockouts:
                symbol_results[symbol] = {
                    "action": "SKIP_LOCKOUT",
                    "active_lockouts": monitor.active_lockouts,
                }
                continue

            runtime.settings = _mainnet_priority_settings(self.settings, symbol)
            leverage = _mainnet_priority_leverage(self.settings, symbol)
            resolved_entry_notional_usdt, sizing = _mainnet_priority_entry_notional_usdt(
                engine,
                fixed_entry_notional_usdt=entry_notional_usdt,
                leverage=leverage,
                entry_margin_fraction=entry_margin_fraction,
            )
            trigger_close_time_ms = orchestrator.latest_runtime_trigger_close_time(
                symbol=symbol,
                timestamp=engine.server_timestamp(),
            )
            timestamp = engine.server_timestamp()
            result = orchestrator.attempt_rule_entry_once(
                symbol=symbol,
                timestamp=timestamp,
                entry_notional_usdt=resolved_entry_notional_usdt,
                leverage=leverage,
                candle_limit=candle_limit,
                trigger_close_time_ms=trigger_close_time_ms,
            )
            symbol_results[symbol] = {
                "sizing": _json_ready(sizing),
                "result": _json_ready(result),
            }
            if result.action == "PLACED":
                monitor_timestamp = engine.server_timestamp()
                monitors[symbol] = orchestrator.reconcile_once(
                    symbol=symbol,
                    timestamp=monitor_timestamp,
                )
                return {
                    "status": "ok",
                    "server_time_ms": monitor_timestamp,
                    "cycle_started_ms": cycle_started_ms,
                    "action": "PLACED",
                    "selected_symbol": symbol,
                    "monitors": _json_ready(monitors),
                    "symbol_results": symbol_results,
                }

        return {
            "status": "ok",
            "server_time_ms": engine.server_timestamp(),
            "cycle_started_ms": cycle_started_ms,
            "action": "NO_TRADE",
            "monitors": _json_ready(monitors),
            "symbol_results": symbol_results,
        }


def serve_dashboard(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    symbol: str = "BTCUSDT",
    symbols: list[str] | None = None,
    environment: str = "testnet",
    controller: DashboardController | None = None,
) -> ThreadingHTTPServer:
    environment = _normalize_environment(environment)
    resolved_symbols = _normalize_symbols(symbols or [symbol])
    default_symbol = resolved_symbols[0]
    controller = controller or DashboardController(
        symbol=default_symbol,
        symbols=resolved_symbols,
        environment=environment,
    )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._write_html(_dashboard_html(default_symbol, environment, resolved_symbols))
                return
            if parsed.path == "/api/summary":
                params = parse_qs(parsed.query)
                target_symbol = params.get("symbol", [default_symbol])[0]
                incident_limit = int(params.get("incident_limit", ["10"])[0])
                display_hours = _query_float(params.get("hours", [None])[0])
                self._write_json(
                    {
                        "status": "ok",
                        "data": controller.summary(
                            symbol=target_symbol,
                            incident_limit=incident_limit,
                            display_hours=display_hours,
                        ),
                    }
                )
                return
            if parsed.path == "/api/service-status":
                self._write_json({"status": "ok", "data": controller.service_status()})
                return
            if parsed.path == "/api/remote-state":
                params = parse_qs(parsed.query)
                target_symbol = params.get("symbol", [default_symbol])[0]
                try:
                    payload = controller.remote_state(symbol=target_symbol)
                except Exception as exc:
                    self._write_json(
                        {"status": "error", "error": str(exc)},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                self._write_json({"status": "ok", "data": payload})
                return
            if parsed.path == "/api/account-status":
                try:
                    payload = controller.account_status()
                except Exception as exc:
                    self._write_json(
                        {"status": "error", "error": str(exc)},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                self._write_json({"status": "ok", "data": payload})
                return
            if parsed.path == "/api/account-equity-history":
                params = parse_qs(parsed.query)
                hours = float(params.get("hours", ["168"])[0])
                max_points = int(params.get("max_points", ["360"])[0])
                try:
                    payload = controller.account_equity_history(
                        hours=hours,
                        max_points=max_points,
                    )
                except Exception as exc:
                    self._write_json(
                        {"status": "error", "error": str(exc)},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                self._write_json({"status": "ok", "data": payload})
                return
            if parsed.path == "/api/account-income-summary":
                params = parse_qs(parsed.query)
                hours = float(params.get("hours", ["168"])[0])
                max_rows = int(params.get("max_rows", ["1000"])[0])
                recent_limit = int(params.get("recent_limit", ["50"])[0])
                refresh = _query_bool(params.get("refresh", ["false"])[0])
                try:
                    payload = controller.account_income_summary(
                        hours=hours,
                        max_rows=max_rows,
                        recent_limit=recent_limit,
                        refresh=refresh,
                    )
                except Exception as exc:
                    self._write_json(
                        {"status": "error", "error": str(exc)},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                self._write_json({"status": "ok", "data": payload})
                return
            self._write_json({"status": "error", "error": "not found"}, status=HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/api/action":
                self._write_json({"status": "error", "error": "not found"}, status=HTTPStatus.NOT_FOUND)
                return
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
            payload = json.loads(raw)
            action = str(payload.get("action", ""))
            try:
                data = controller.handle_action(action=action, payload=payload)
            except Exception as exc:
                self._write_json(
                    {"status": "error", "error": str(exc)},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            self._write_json({"status": "ok", "data": data})

        def _write_json(self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
            encoded = json.dumps(_json_ready(payload), indent=2).encode("utf-8")
            try:
                self.send_response(status.value)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
            except BrokenPipeError:
                return

        def _write_html(self, html: str) -> None:
            encoded = html.encode("utf-8")
            try:
                self.send_response(HTTPStatus.OK.value)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
            except BrokenPipeError:
                return

    return ThreadingHTTPServer((host, port), Handler)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, Decimal):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return _json_ready(asdict(value))
    return value


def _account_usdt_status_from_account(account: dict[str, Any], *, server_time_ms: int) -> dict[str, Any]:
    usdt_asset: dict[str, Any] = {}
    assets = account.get("assets")
    if isinstance(assets, list):
        for asset in assets:
            if isinstance(asset, dict) and str(asset.get("asset", "")).upper() == "USDT":
                usdt_asset = asset
                break

    return {
        "asset": "USDT",
        "wallet_balance_usdt": _float_or_none(
            usdt_asset.get("walletBalance", account.get("totalWalletBalance"))
        ),
        "available_balance_usdt": _float_or_none(
            usdt_asset.get("availableBalance", account.get("availableBalance"))
        ),
        "margin_balance_usdt": _float_or_none(
            usdt_asset.get("marginBalance", account.get("totalMarginBalance"))
        ),
        "unrealized_pnl_usdt": _float_or_none(
            usdt_asset.get("unrealizedProfit", account.get("totalUnrealizedProfit"))
        ),
        "initial_margin_usdt": _float_or_none(
            usdt_asset.get("initialMargin", account.get("totalInitialMargin"))
        ),
        "maint_margin_usdt": _float_or_none(
            usdt_asset.get("maintMargin", account.get("totalMaintMargin"))
        ),
        "cross_wallet_balance_usdt": _float_or_none(usdt_asset.get("crossWalletBalance")),
        "max_withdraw_amount_usdt": _float_or_none(usdt_asset.get("maxWithdrawAmount")),
        "server_time_ms": server_time_ms,
    }


def _account_equity_points_from_rows(rows: list[tuple[str, int]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for payload_json, recorded_at_ms in rows:
        try:
            payload = json.loads(payload_json)
        except json.JSONDecodeError:
            continue
        usdt = payload.get("usdt")
        if not isinstance(usdt, dict):
            continue
        wallet = _float_or_none(usdt.get("wallet_balance_usdt"))
        margin = _float_or_none(usdt.get("margin_balance_usdt"))
        available = _float_or_none(usdt.get("available_balance_usdt"))
        unrealized = _float_or_none(usdt.get("unrealized_pnl_usdt"))
        equity = margin
        if equity is None and wallet is not None:
            equity = wallet + (unrealized or 0.0)
        if equity is None:
            continue
        points.append(
            {
                "recorded_at_ms": int(recorded_at_ms),
                "equity_usdt": equity,
                "wallet_balance_usdt": wallet,
                "margin_balance_usdt": margin,
                "available_balance_usdt": available,
                "unrealized_pnl_usdt": unrealized,
            }
        )
    return points


def _account_equity_summary(points: list[dict[str, Any]]) -> dict[str, Any]:
    if not points:
        return {
            "first_recorded_at_ms": None,
            "last_recorded_at_ms": None,
            "start_equity_usdt": None,
            "current_equity_usdt": None,
            "change_usdt": None,
            "change_pct": None,
            "high_equity_usdt": None,
            "low_equity_usdt": None,
        }
    start = float(points[0]["equity_usdt"])
    current = float(points[-1]["equity_usdt"])
    change = current - start
    return {
        "first_recorded_at_ms": points[0]["recorded_at_ms"],
        "last_recorded_at_ms": points[-1]["recorded_at_ms"],
        "start_equity_usdt": start,
        "current_equity_usdt": current,
        "change_usdt": change,
        "change_pct": (change / start) * 100.0 if start else None,
        "high_equity_usdt": max(float(point["equity_usdt"]) for point in points),
        "low_equity_usdt": min(float(point["equity_usdt"]) for point in points),
    }


def _account_income_start_time_ms(
    runtime: TestnetExecutionRuntime,
    *,
    hours: float,
    end_time_ms: int,
) -> int | None:
    fallback = end_time_ms - int(hours * 3_600_000) if hours > 0 else None
    runtime.initialize_storage()
    query = """
        SELECT recorded_at_ms
        FROM account_snapshots
        WHERE symbol = 'ACCOUNT'
          AND snapshot_type = 'rest_account_usdt_balance'
    """
    params: list[Any] = []
    if fallback is not None:
        query += " AND recorded_at_ms >= ?"
        params.append(fallback)
    query += " ORDER BY recorded_at_ms ASC LIMIT 1"
    with _runtime_db_connection(runtime.database_path) as conn:
        row = conn.execute(query, params).fetchone()
    if row is not None:
        return int(row[0])
    return fallback


def _fetch_income_history(
    client: BinanceFuturesTestnetClient,
    *,
    timestamp: int,
    start_time_ms: int | None,
    end_time_ms: int,
    max_rows: int,
) -> list[dict[str, Any]]:
    max_rows = max(1, min(int(max_rows), 10_000))
    rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, ...]] = set()
    cursor = start_time_ms
    while len(rows) < max_rows:
        limit = min(1000, max_rows - len(rows))
        batch = client.query_income_history(
            timestamp=timestamp,
            start_time=cursor,
            end_time=end_time_ms,
            limit=limit,
        )
        if not isinstance(batch, list) or not batch:
            break
        for item in batch:
            if not isinstance(item, dict):
                continue
            key = _income_row_key(item)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            rows.append(dict(item))
        if len(batch) < limit:
            break
        batch_times = [
            value
            for value in (_int_or_none(item.get("time")) for item in batch if isinstance(item, dict))
            if value is not None
        ]
        if not batch_times:
            break
        next_cursor = max(batch_times) + 1
        if cursor is not None and next_cursor <= cursor:
            break
        cursor = next_cursor
        if cursor > end_time_ms:
            break
    return rows


def _income_row_key(row: dict[str, Any]) -> tuple[str, ...]:
    return (
        str(row.get("incomeType") or ""),
        str(row.get("tranId") or row.get("trandId") or ""),
        str(row.get("tradeId") or ""),
        str(row.get("symbol") or ""),
        str(row.get("income") or ""),
        str(row.get("time") or ""),
    )


def _normalize_income_history_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        income = _float_or_none(row.get("income"))
        time_ms = _int_or_none(row.get("time"))
        if income is None or time_ms is None:
            continue
        normalized.append(
            {
                "symbol": str(row.get("symbol") or "").upper(),
                "income_type": str(row.get("incomeType") or "").upper(),
                "income_usdt": income,
                "asset": str(row.get("asset") or "").upper(),
                "info": str(row.get("info") or ""),
                "time_ms": time_ms,
                "tran_id": str(row.get("tranId") or row.get("trandId") or ""),
                "trade_id": str(row.get("tradeId") or ""),
            }
        )
    normalized.sort(key=lambda item: (int(item["time_ms"]), item["income_type"], item["tran_id"]))
    return normalized


def _account_income_summary(
    rows: list[dict[str, Any]],
    *,
    symbols: list[str],
) -> dict[str, Any]:
    usdt_rows = [row for row in rows if row.get("asset") in {"", "USDT"}]
    totals_by_type: dict[str, float] = {}
    for row in usdt_rows:
        income_type = str(row.get("income_type") or "UNKNOWN")
        totals_by_type[income_type] = totals_by_type.get(income_type, 0.0) + float(row["income_usdt"])
    realized = totals_by_type.get("REALIZED_PNL", 0.0)
    commission = totals_by_type.get("COMMISSION", 0.0)
    funding = totals_by_type.get("FUNDING_FEE", 0.0)
    tracked_types = {"REALIZED_PNL", "COMMISSION", "FUNDING_FEE"}
    trading_net = realized + commission + funding
    total_income = sum(totals_by_type.values())
    by_symbol = {
        symbol: _income_symbol_summary(usdt_rows, symbol=symbol)
        for symbol in symbols
    }
    return {
        "row_count": len(usdt_rows),
        "tracked_row_count": sum(1 for row in usdt_rows if row.get("income_type") in tracked_types),
        "first_income_time_ms": min((int(row["time_ms"]) for row in usdt_rows), default=None),
        "last_income_time_ms": max((int(row["time_ms"]) for row in usdt_rows), default=None),
        "realized_pnl_usdt": realized,
        "commission_usdt": commission,
        "funding_fee_usdt": funding,
        "trading_net_income_usdt": trading_net,
        "other_income_usdt": total_income - trading_net,
        "total_income_usdt": total_income,
        "totals_by_type": totals_by_type,
        "by_symbol": by_symbol,
    }


def _income_symbol_summary(rows: list[dict[str, Any]], *, symbol: str) -> dict[str, Any]:
    symbol_rows = [row for row in rows if row.get("symbol") == symbol]
    totals_by_type: dict[str, float] = {}
    for row in symbol_rows:
        income_type = str(row.get("income_type") or "UNKNOWN")
        totals_by_type[income_type] = totals_by_type.get(income_type, 0.0) + float(row["income_usdt"])
    realized = totals_by_type.get("REALIZED_PNL", 0.0)
    commission = totals_by_type.get("COMMISSION", 0.0)
    funding = totals_by_type.get("FUNDING_FEE", 0.0)
    return {
        "row_count": len(symbol_rows),
        "realized_pnl_usdt": realized,
        "commission_usdt": commission,
        "funding_fee_usdt": funding,
        "trading_net_income_usdt": realized + commission + funding,
        "totals_by_type": totals_by_type,
    }


def _downsample_points(points: list[dict[str, Any]], *, max_points: int) -> list[dict[str, Any]]:
    if len(points) <= max_points:
        return points
    if max_points <= 2:
        return [points[0], points[-1]]
    last_index = len(points) - 1
    indexes = {
        round(index * last_index / (max_points - 1))
        for index in range(max_points)
    }
    return [points[index] for index in sorted(indexes)]


def _utc_now_ms() -> int:
    return int(time.time() * 1000)


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _query_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _query_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _display_since_ms(hours: float | None) -> int | None:
    if hours is None or hours <= 0:
        return None
    return _utc_now_ms() - int(hours * 3_600_000)


def _mainnet_priority_settings(settings: Settings, symbol: str) -> Settings:
    exit_policy = settings.exit_policy
    fixed_take_profit_r = settings.fixed_take_profit_r
    if symbol.upper() in {"ETHUSDT", "SOLUSDT"}:
        exit_policy = "time_stop_only"
        fixed_take_profit_r = 1.0
    return replace(
        settings,
        strategy_mode="multi_symbol_priority_v1",
        exit_policy=exit_policy,
        fixed_take_profit_r=fixed_take_profit_r,
    )


def _mainnet_priority_leverage(settings: Settings, symbol: str) -> int:
    return settings.live_start_leverage


def _mainnet_entry_notional_usdt(
    engine: TestnetExecutionEngine,
    *,
    fixed_entry_notional_usdt: float | None,
    leverage: int,
    entry_margin_fraction: float,
) -> tuple[float, dict[str, object]]:
    if fixed_entry_notional_usdt is not None:
        raise ValueError("entry_notional_usdt is disabled on mainnet; use entry_margin_fraction")
    if not 0.0 < entry_margin_fraction <= 1.0:
        raise ValueError("entry_margin_fraction must be > 0 and <= 1")

    timestamp = engine.server_timestamp()
    account = engine.client.query_account(timestamp=timestamp)
    available_balance = _account_available_usdt(account)
    if available_balance is None:
        raise ValueError("account availableBalance is missing; cannot size priority entry")
    entry_notional = available_balance * entry_margin_fraction * leverage
    return entry_notional, {
        "mode": "available_balance_fraction",
        "available_balance_usdt": available_balance,
        "entry_margin_fraction": entry_margin_fraction,
        "leverage": leverage,
        "entry_notional_usdt": entry_notional,
        "server_time_ms": timestamp,
    }


def _mainnet_priority_entry_notional_usdt(
    engine: TestnetExecutionEngine,
    *,
    fixed_entry_notional_usdt: float | None,
    leverage: int,
    entry_margin_fraction: float,
) -> tuple[float, dict[str, object]]:
    return _mainnet_entry_notional_usdt(
        engine,
        fixed_entry_notional_usdt=fixed_entry_notional_usdt,
        leverage=leverage,
        entry_margin_fraction=entry_margin_fraction,
    )


def _account_available_usdt(account: dict[str, Any]) -> float | None:
    assets = account.get("assets")
    if isinstance(assets, list):
        for asset in assets:
            if isinstance(asset, dict) and str(asset.get("asset", "")).upper() == "USDT":
                value = asset.get("availableBalance")
                if value not in (None, ""):
                    return float(value)
    return _float_or_none(account.get("availableBalance"))


def _remote_has_open_position(monitor: Any) -> bool:
    if monitor is None or monitor.reconciliation is None:
        return False
    return any(
        abs(float(position.get("positionAmt", "0"))) > 1e-9
        for position in monitor.reconciliation.remote_state.positions
    )


def _remote_account_open_position_symbols(engine: TestnetExecutionEngine) -> list[str]:
    positions = engine.client.query_position_risk(timestamp=engine.server_timestamp())
    symbols: list[str] = []
    for position in positions:
        if abs(float(position.get("positionAmt", "0"))) > 1e-9:
            symbol = str(position.get("symbol", "")).upper()
            if symbol and symbol not in symbols:
                symbols.append(symbol)
    return symbols


def _account_position_symbol(runtime: TestnetExecutionRuntime, symbols: list[str]) -> str | None:
    for symbol in symbols:
        if (
            runtime.load_managed_trade_state(symbol) is not None
            or runtime.load_expected_state(symbol) is not None
        ):
            return symbol
    for symbol in symbols:
        snapshot = runtime.latest_account_snapshot(symbol)
        remote_state = None
        if snapshot is not None:
            payload = snapshot.get("payload")
            if isinstance(payload, dict):
                remote_state = payload.get("remote_state")
        if isinstance(remote_state, dict) and _remote_state_has_position(remote_state):
            return symbol
    return None


def _remote_state_has_position(remote_state: dict[str, Any]) -> bool:
    positions = remote_state.get("positions")
    if not isinstance(positions, list):
        return False
    return any(
        isinstance(position, dict)
        and abs(float(position.get("positionAmt", "0") or 0.0)) > 1e-9
        for position in positions
    )


def _managed_symbol(runtime: TestnetExecutionRuntime, symbols: list[str]) -> str | None:
    for symbol in symbols:
        if runtime.load_managed_trade_state(symbol) is not None:
            return symbol
    return None


def _launchctl_status(label: str) -> dict[str, Any]:
    command = ["launchctl", "print", f"gui/{os.getuid()}/{label}"]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=3.0,
        )
    except Exception as exc:
        return {
            "label": label,
            "loaded": False,
            "state": "unknown",
            "pid": None,
            "runs": None,
            "error": str(exc),
        }
    if completed.returncode != 0:
        return {
            "label": label,
            "loaded": False,
            "state": "unloaded",
            "pid": None,
            "runs": None,
            "error": completed.stderr.strip() or completed.stdout.strip(),
        }
    output = completed.stdout
    return {
        "label": label,
        "loaded": True,
        "state": _launchctl_value(output, "state") or "unknown",
        "pid": _int_or_none(_launchctl_value(output, "pid")),
        "runs": _int_or_none(_launchctl_value(output, "runs")),
        "path": _launchctl_value(output, "path"),
        "program": _launchctl_value(output, "program"),
    }


def _launchctl_value(output: str, key: str) -> str | None:
    prefix = f"{key} = "
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return None


def _int_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(str(value))
    except ValueError:
        return None


def _dashboard_html(default_symbol: str, environment: str, symbols: list[str] | None = None) -> str:
    template = _DASHBOARD_TEMPLATE_PATH.read_text(encoding="utf-8")
    environment = _normalize_environment(environment)
    environment_label = "MAINNET LIVE" if environment == "mainnet" else "TESTNET"
    symbols_json = json.dumps(_normalize_symbols(symbols or [default_symbol])).replace("<", "\\u003c")
    return (
        template
        .replace("__DEFAULT_SYMBOL__", html.escape(default_symbol, quote=True))
        .replace("__SYMBOLS_JSON__", symbols_json)
        .replace("__ENVIRONMENT__", html.escape(environment, quote=True))
        .replace("__ENVIRONMENT_LABEL__", html.escape(environment_label, quote=True))
    )


def _normalize_environment(environment: str) -> str:
    normalized = environment.strip().lower()
    if normalized not in {"testnet", "mainnet"}:
        raise ValueError("dashboard environment must be one of ['testnet', 'mainnet']")
    return normalized


def _normalize_symbols(symbols: list[str]) -> list[str]:
    normalized: list[str] = []
    for symbol in symbols:
        value = str(symbol).strip().upper()
        if value and value not in normalized:
            normalized.append(value)
    if not normalized:
        raise ValueError("dashboard requires at least one symbol")
    return normalized
